<template>
  <div class="llm-chat">
    <div>
      <input
        v-model="username"
        placeholder="Username"
        class="input"
        :disabled="wsConnected || loading"
        style="margin-right: 8px;"
      />
      <button @click="getToken" :disabled="wsConnected || !username || loading">
        Get Token
      </button>
      <span v-if="token" class="token-display">Token: {{ token }}</span>
    </div>
    <div style="margin-top: 8px;">
      <button v-if="wsConnected" @click="disconnectWS" style="color:red;">Disconnect</button>
      <span v-if="wsConnected" style="color: green; margin-left: 1em;">Connected</span>
    </div>
    <form @submit.prevent="sendQuery" style="margin-top: 12px;">
      <input
        v-model="userInput"
        placeholder="Ask something..."
        class="input"
        :disabled="!wsConnected || loading"
      />
      <button :disabled="!wsConnected || loading || !userInput">Send</button>
    </form>
    <div class="response">
      <pre v-if="streaming">{{ streamedResponse }}</pre>
      <pre v-else>{{ response }}</pre>
      <pre v-if="error" style="color:red;">{{ error }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref, onBeforeUnmount } from 'vue';

const userInput = ref('');
const response = ref('');
const streamedResponse = ref('');
const loading = ref(false);
const streaming = ref(false);
const username = ref('');
const token = ref('');
const error = ref('');
const wsConnected = ref(false);

let ws = null;

const API_BASE = import.meta.env.DEV
  ? 'http://localhost:5990/api'
  : '/api';

const WS_URL = import.meta.env.DEV
  ? 'ws://localhost:5990/ws'
  : `wss://${window.location.host}/ws`;

async function getToken() {
  error.value = '';
  token.value = '';
  wsConnected.value = false;
  loading.value = true;
  try {
    const res = await fetch(`${API_BASE}/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username.value }),
    });
    const data = await res.json();
    if (data.token) {
      token.value = data.token;
      await connectWS(); // Immediately connect after getting token
    } else {
      error.value = data.error || "Failed to get token";
    }
  } catch (e) {
    error.value = "Failed to fetch token";
  } finally {
    loading.value = false;
  }
}

function connectWS() {
  return new Promise((resolve, reject) => {
    error.value = '';
    if (ws) ws.close();
    ws = new WebSocket(`${WS_URL}?token=${token.value}`);
    ws.onopen = () => {
      wsConnected.value = true;
      resolve();
    };
    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'llm_stream') {
        streamedResponse.value += msg.data;
        streaming.value = true;
      } else if (msg.type === 'llm_end') {
        streaming.value = false;
        response.value = streamedResponse.value;
      } else if (msg.type === 'error') {
        error.value = msg.data;
        streaming.value = false;
      }
    };
    ws.onerror = (e) => {
      error.value = "WebSocket error.";
      wsConnected.value = false;
      streaming.value = false;
      reject(e);
    };
    ws.onclose = () => {
      ws = null;
      wsConnected.value = false;
      streaming.value = false;
    };
  });
}

function sendQuery() {
  if (!wsConnected.value || !ws || ws.readyState !== 1) {
    error.value = "WebSocket not connected.";
    return;
  }
  streamedResponse.value = '';
  streaming.value = true;
  response.value = '';
  error.value = '';
  ws.send(JSON.stringify({ action: 'query', data: userInput.value }));
}

function disconnectWS() {
  if (ws) ws.close();
  wsConnected.value = false;
}

onBeforeUnmount(() => {
  if (ws) ws.close();
});
</script>

<style scoped>
.llm-chat { width:600px; max-width: 80%; margin: 0 auto; }
.input { width: 180px; padding: 8px; margin: 4px 0; }
.token-display { font-size: 0.85em; color: #357; margin-left: 1em; }

.response {
  margin-top: 20px;
  min-height: 120px;
  max-height: 350px;
  background: #eee;
  padding: 12px;
  border-radius: 6px;
  text-align: left;
  overflow-y: auto;
  word-break: break-word;
  white-space: pre-wrap;
  width:100%;
}

.response pre {
  white-space: pre-wrap !important;
  word-break: break-all !important; /* ← this is the key! */
  text-align: left;
}

</style>
