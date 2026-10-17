'use strict';

// This frontend uses the existing Python /api/status and /api/generate endpoints.
// No external API keys, cloud inference endpoints or web frameworks are needed.
const $ = (id) => document.getElementById(id);
const form = $('chatForm');
const field = $('messageInput');
const send = $('sendButton');
const model = $('modelSelect');
const reasoning = $('reasoningSelect');
const task = $('taskSelect');
const welcome = $('welcome');
const messages = $('messages');
const conversation = $('conversation');
const notice = $('errorMessage');
const statusText = $('statusText');
const statusDot = $('statusDot');
const newChat = $('newChat');
const suggestions = document.querySelectorAll('.suggestion');

let busy = false;
let hasAdapter = false;
let turns = [];

function setStatus(text, level = '') {
  statusText.textContent = text;
  statusDot.className = 'status-indicator' + (level ? ' ' + level : '');
}
function setError(message) {
  notice.textContent = message;
  notice.hidden = !message;
}
function scrollToBottom() {
  conversation.scrollTop = conversation.scrollHeight;
}
function setBusy(value) {
  busy = value;
  send.disabled = value;
  newChat.disabled = value;
  model.disabled = value;
  task.disabled = value;
  reasoning.disabled = value;
  for (const suggestion of suggestions) suggestion.disabled = value;
  send.setAttribute('aria-label', value ? 'Generating response' : 'Send message');
}
function adjustHeight() {
  field.style.height = 'auto';
  field.style.height = Math.min(field.scrollHeight, 152) + 'px';
}
function addMessage(role, text, meta = '', css = '') {
  const wrapper = document.createElement('article');
  wrapper.className = 'message' + (role === 'You' ? ' mine' : '') + (css ? ' ' + css : '');
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrapper.append(label, bubble);
  if (meta) {
    const subtitle = document.createElement('div');
    subtitle.className = 'message-meta';
    subtitle.textContent = meta;
    wrapper.append(subtitle);
  }
  messages.append(wrapper);
  welcome.hidden = true;
  scrollToBottom();
  return { wrapper, bubble, label };
}
function resetConversation() {
  if (busy) return;
  turns = [];
  messages.replaceChildren();
  welcome.hidden = false;
  field.value = '';
  setError('');
  adjustHeight();
  field.focus();
}

async function getStatus() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('Server unavailable');
    const info = await response.json();
    hasAdapter = Boolean(info.adapter_available);
    const tuned = model.querySelector('option[value="tuned"]');
    tuned.disabled = !hasAdapter;
    tuned.textContent = hasAdapter ? 'Fine-tuned Qwen (LoRA)' : 'Fine-tuned (not trained yet)';
    // Default to the trained model when available, otherwise use the base model.
    model.value = hasAdapter ? 'tuned' : 'base';
    setStatus(hasAdapter ? 'LoRA adapter found · Model loads when you send' :
      'Original model ready to test · Fine-tuned adapter not found', hasAdapter ? 'ok' : 'warn');
  } catch (error) {
    model.value = 'base';
    model.querySelector('option[value="tuned"]').disabled = true;
    setStatus('Cannot connect to local Python server', 'warn');
    setError('Start the server in Command Prompt with: python web_app.py');
  }
}

async function sendMessage() {
  if (busy) return;
  const question = field.value.trim();
  if (!question) return field.focus();
  setError('');
  const choice = model.value;
  const mode = reasoning.value;
  const questionType = task.value;
  const label = choice === 'tuned' ? 'Fine-tuned Qwen' : 'Original Qwen';
  const modeLabel = mode === 'on' ? 'Thinking on' : 'Thinking off';
  const history = turns.slice(-1); // Keep one previous exchange for the 512-token CPU context.
  addMessage('You', question);
  field.value = '';
  adjustHeight();
  const pending = addMessage(label, 'Generating on your CPU...');
  setBusy(true);
  const started = Date.now();
  const ticker = window.setInterval(() => {
    pending.bubble.textContent = 'Generating on your CPU... (' + Math.round((Date.now() - started) / 1000) + 's)';
  }, 1000);
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode, task: questionType, choice,
        max_new_tokens: 192, history })
    });
    let data;
    try { data = await response.json(); }
    catch { throw new Error('Server returned an unreadable response.'); }
    if (!response.ok) throw new Error(data.error || 'The local model could not generate a response.');
    const answer = data.outputs && data.outputs[choice];
    if (!answer) throw new Error('The model returned no answer for the selected model.');
    await window.ChatFormatter.renderModelMessage(
      pending.bubble, answer.text || '(Model generated no visible text.)'
    );
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = modeLabel + ' · ' + Number(answer.seconds || 0).toFixed(1) + 's · ' + Number(answer.output_tokens || 0) + ' tokens';
    pending.wrapper.append(meta);
    turns.push({ user: question, assistant: answer.text || '' });
    if (turns.length > 3) turns = turns.slice(-3);
    setStatus(label + ' · ' + modeLabel + ' · Ready for another question', 'ok');
  } catch (error) {
    pending.wrapper.classList.add('error');
    pending.label.textContent = 'Could not generate response';
    pending.bubble.textContent = error.message;
    setStatus('Check error above or the Command Prompt window', 'warn');
    setError('No model weights were changed. You may try a shorter question or start a new chat.');
  } finally {
    window.clearInterval(ticker);
    setBusy(false);
    scrollToBottom();
    field.focus();
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); sendMessage(); });
field.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage();
  }
});
field.addEventListener('input', adjustHeight);
newChat.addEventListener('click', resetConversation);
for (const control of [model, reasoning, task]) {
  control.addEventListener('change', () => {
    // Previous assistant replies were produced with different settings.
    // Starting fresh avoids comparing responses using an incompatible history.
    turns = [];
    messages.replaceChildren();
    welcome.hidden = false;
    setError('');
    field.focus();
    setStatus('Chat reset for your new settings', 'ok');
  });
}
for (const suggestion of suggestions) {
  suggestion.addEventListener('click', () => {
    if (task.value !== suggestion.dataset.task) {
      task.value = suggestion.dataset.task;
      turns = [];
      messages.replaceChildren();
      welcome.hidden = false;
    }
    field.value = suggestion.dataset.question || '';
    adjustHeight();
    field.focus();
  });
}
getStatus();
adjustHeight();
