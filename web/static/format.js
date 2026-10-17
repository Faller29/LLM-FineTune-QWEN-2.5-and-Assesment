'use strict';

// Safe, dependency-free Markdown subset. All model text goes through textContent:
// HTML returned by the model is NEVER trusted as markup.
// MathJax renders TeX delimiters after the DOM nodes have been constructed.
(function () {
  const MATH_OPENERS = [
    ['\\(', '\\)'], ['\\[', '\\]'], ['$$', '$$'], ['$', '$']
  ];

  function appendPlainMarkdown(parent, input) {
    // Small Markdown subset: **bold**, *italic*, `code`. No raw HTML/links/images.
    const marker = /(\*\*[^*\n]+\*\*|__[^_\n]+__|`[^`\n]+`|\*[^*\n]+\*)/g;
    let previous = 0;
    for (const found of input.matchAll(marker)) {
      const index = found.index;
      if (index > previous) parent.append(document.createTextNode(input.slice(previous, index)));
      const token = found[0];
      const isCode = token[0] === '`';
      const isBold = token.startsWith('**') || token.startsWith('__');
      const element = document.createElement(isCode ? 'code' : (isBold ? 'strong' : 'em'));
      element.textContent = token.slice(isBold ? 2 : 1, isBold ? -2 : -1);
      parent.append(element);
      previous = index + token.length;
    }
    if (previous < input.length) parent.append(document.createTextNode(input.slice(previous)));
  }

  function findMathToken(text, fromIndex) {
    for (let index = fromIndex; index < text.length; index++) {
      // NVIDIA-style final answers may contain a bare \boxed{...} rather than
      // explicit \( ... \) delimiters. Preserve the raw answer, but wrap it
      // in inline math ONLY for display (never change the saved model output).
      if (text.startsWith('\\boxed{', index)) {
        let depth = 0;
        for (let cursor = index + 6; cursor < text.length; cursor++) {
          if (text[cursor] === '{') depth++;
          if (text[cursor] === '}') {
            depth--;
            if (depth === 0) return { start: index, end: cursor + 1, bareBox: true };
          }
        }
      }
      for (const [opening, closing] of MATH_OPENERS) {
        if (!text.startsWith(opening, index)) continue;
        // Currency like "$5" should not be mistaken for math without a closing $.
        if (opening === '$' && (text[index + 1] === '$' || /\s/.test(text[index + 1] || ''))) continue;
        const end = text.indexOf(closing, index + opening.length);
        if (end < 0) continue;
        if (opening === '$' && (end === index + 1 || /\s/.test(text[end - 1]))) continue;
        return { start: index, end: end + closing.length };
      }
    }
    return null;
  }

  function appendInline(parent, text) {
    let start = 0;
    while (start < text.length) {
      const token = findMathToken(text, start);
      if (!token) {
        appendPlainMarkdown(parent, text.slice(start));
        break;
      }
      if (token.start > start) appendPlainMarkdown(parent, text.slice(start, token.start));
      const math = document.createElement('span');
      math.className = 'math-fragment';
      const literal = text.slice(token.start, token.end);
      math.textContent = token.bareBox ? '\\(' + literal + '\\)' : literal;
      parent.append(math);
      start = token.end;
    }
  }

  function addParagraph(parent, lines) {
    if (!lines.length) return;
    const paragraph = document.createElement('p');
    appendInline(paragraph, lines.join('\n'));
    parent.append(paragraph);
    lines.length = 0;
  }

  function renderMarkdown(container, source) {
    container.replaceChildren();
    container.classList.add('formatted');
    const lines = String(source || '').replace(/\r\n?/g, '\n').split('\n');
    const paragraph = [];
    let currentList = null;
    let codeBuffer = null;
    let fenceMarker = '';

    function closeList() { currentList = null; }
    function closeCode() {
      if (!codeBuffer) return;
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = codeBuffer.join('\n');
      pre.append(code);
      container.append(pre);
      codeBuffer = null;
    }

    for (const line of lines) {
      const fence = line.match(/^\s{0,3}(`{3,}|~{3,})(?:\S*)?\s*$/);
      if (codeBuffer) {
        if (fence && fence[1][0] === fenceMarker[0] && fence[1].length >= fenceMarker.length) closeCode();
        else codeBuffer.push(line);
        continue;
      }
      if (fence) {
        addParagraph(container, paragraph);
        closeList();
        codeBuffer = [];
        fenceMarker = fence[1];
        continue;
      }
      if (!line.trim()) {
        addParagraph(container, paragraph);
        closeList();
        continue;
      }
      const heading = line.match(/^\s{0,3}(#{1,4})\s+(.*)$/);
      if (heading) {
        addParagraph(container, paragraph);
        closeList();
        const node = document.createElement('h' + Math.min(heading[1].length + 2, 6));
        appendInline(node, heading[2]);
        container.append(node);
        continue;
      }
      const bullet = line.match(/^\s{0,3}(?:([-*+])\s+|(\d+)\.\s+)(.*)$/);
      if (bullet) {
        addParagraph(container, paragraph);
        const kind = bullet[2] ? 'ol' : 'ul';
        if (!currentList || currentList.tagName.toLowerCase() !== kind) {
          currentList = document.createElement(kind);
          container.append(currentList);
        }
        const item = document.createElement('li');
        appendInline(item, bullet[3]);
        currentList.append(item);
        continue;
      }
      closeList();
      paragraph.push(line);
    }
    addParagraph(container, paragraph);
    closeCode();
  }

  async function renderModelMessage(container, text) {
    renderMarkdown(container, text);
    if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
      try {
        await window.MathJax.typesetPromise([container]);
      } catch (error) {
        console.warn('MathJax could not render this response; original TeX remains readable.', error);
      }
    }
  }

  window.ChatFormatter = { renderModelMessage, renderMarkdown };
})();
