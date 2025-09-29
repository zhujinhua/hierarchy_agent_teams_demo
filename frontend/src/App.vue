<template>
  <div class="app-wrapper">
    <div class="chat-container">
      <!-- Title Bar -->
      <div class="chat-header">
        <h1>What's on the agenda today?</h1>
      </div>

      <!-- Chat Messages Area -->
      <div ref="chatBox" class="chat-messages">
        <div
          v-for="(m, i) in filteredMessages"
          :key="i"
          class="message-row"
          :class="m.role === 'user' ? 'message-row--user' : 'message-row--assistant'"
        >
          <div
            class="message-bubble"
            :class="m.role === 'user' ? 'message-bubble--user' : 'message-bubble--assistant'"
            v-html="formatMessage(m.content)"
          >
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="input-container">
        <textarea
          v-model="prompt"
          rows="1"
          class="input-field"
          placeholder="Ask anything"
          :disabled="streaming"
          @keydown.enter.exact.prevent="start"
        ></textarea>
        <button
          @click="start"
          :disabled="streaming || !prompt.trim()"
          class="send-btn"
        >
          {{ streaming ? "Streaming..." : "Send" }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, nextTick, watch, computed } from "vue";
// Fixed import syntax for marked (uses named export)
import { marked } from "marked";

export default {
  setup() {
    const prompt = ref("");
    const messages = ref([]);
    const streaming = ref(false);
    const chatBox = ref(null);
    let ws;
    let assistantMessageIndex = -1;

    console.log("Chat component initialized");

    // Computed property to filter out empty messages
    const filteredMessages = computed(() => {
      return messages.value.filter((message) => message.content.trim() !== "");
    });

    watch(filteredMessages, () => {
      console.log(`Filtered messages updated. Total messages: ${filteredMessages.value.length}`);
      scrollToBottom();
    });

    const scrollToBottom = () => {
      nextTick(() => {
        if (chatBox.value) {
          chatBox.value.scrollTop = chatBox.value.scrollHeight;
          console.log("Scrolled to bottom of chat");
        } else {
          console.warn("Chat box element not found for scrolling");
        }
      });
    };

    // Format messages with Markdown (supports **bold**, *italic*, etc.)
    const formatMessage = (content) => {
      return marked.parse(content);
    };

    const start = () => {
      if (streaming.value) {
        console.log("Cannot send message - already streaming");
        return;
      }

      if (!prompt.value.trim()) {
        console.log("Cannot send empty message");
        return;
      }

      console.log("Starting new message send");
      streaming.value = true;

      const userMessageContent = prompt.value.trim();
      if (userMessageContent) {
        const userMessage = {
          role: "user",
          content: userMessageContent,
          isNew: true,
        };
        messages.value.push(userMessage);
        console.log(`Added user message: "${userMessageContent.substring(0, 50)}${userMessageContent.length > 50 ? '...' : ''}"`);

        setTimeout(() => {
          userMessage.isNew = false;
        }, 200);
      }

      const userPrompt = prompt.value;
      prompt.value = "";
      console.log("Cleared input field");

      console.log("Connecting to WebSocket at ws://localhost:8000/ws/stream");
      ws = new WebSocket("ws://localhost:8000/ws/stream");

      ws.onopen = () => {
        console.log("WebSocket connection established");
        try {
          const payload = JSON.stringify({ prompt: userPrompt });
          console.log(`Sending payload: ${payload.substring(0, 100)}${payload.length > 100 ? '...' : ''}`);
          ws.send(payload);
        } catch (error) {
          console.error("Error sending payload to WebSocket:", error);
          handleWebSocketError("Failed to send your message. Please try again.");
        }
      };

      ws.onmessage = (ev) => {
        console.log("Received WebSocket message");
        try {
          const msg = JSON.parse(ev.data);
          console.log(`Message event: ${msg.event}`, msg);

          if (msg.event === "token") {
            const tokenContent = msg.data.trim();
            if (tokenContent) {
              if (assistantMessageIndex === -1) {
                messages.value.push({
                  role: "assistant",
                  content: tokenContent,
                  isNew: true,
                });
                assistantMessageIndex = messages.value.length - 1;
                console.log("Created new assistant message");

                setTimeout(() => {
                  if (messages.value[assistantMessageIndex]) {
                    messages.value[assistantMessageIndex].isNew = false;
                  }
                }, 200);
              } else {
                messages.value[assistantMessageIndex].content += tokenContent;
                console.log(`Appended token. Current assistant message length: ${messages.value[assistantMessageIndex].content.length}`);
              }
            }
          } else if (msg.event === "done") {
            console.log("Received 'done' event from server");
            streaming.value = false;
            assistantMessageIndex = -1;
          } else if (msg.event === "error") {
            console.error("Server returned an error:", msg.data);
            handleWebSocketError(msg.data || "An error occurred while processing your request.");
          } else {
            console.log(`Received unknown event type: ${msg.event}`);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
          console.error("Raw message data:", ev.data);
          handleWebSocketError("Failed to parse the server's response. Please try again.");
        }
      };

      ws.onclose = (event) => {
        console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason}`);
        streaming.value = false;
        assistantMessageIndex = -1;
        if (event.code !== 1000) {
          handleWebSocketError(`Connection closed unexpectedly (Code: ${event.code}, Reason: ${event.reason})`);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error occurred:", error);
        handleWebSocketError("A WebSocket error occurred. Please try again.");
      };
    };

    const handleWebSocketError = (errorMessage) => {
      streaming.value = false;
      assistantMessageIndex = -1;
      const errorMessageObj = {
        role: "assistant",
        content: errorMessage,
      };
      if (errorMessageObj.content.trim()) {
        messages.value.push(errorMessageObj);
      }
    };

    return { prompt, filteredMessages, streaming, start, chatBox, formatMessage };
  },
};
</script>

<style scoped>
/* --------------------------
Global Wrapper (Full Screen)
-------------------------- */
.app-wrapper {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f3f4f6;
  padding: 1rem;
}

/* --------------------------
Chat Container (Card Style)
-------------------------- */
.chat-container {
  width: 100%;
  max-width: 600px;
  height: 80vh;
  background-color: #ffffff;
  border-radius: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* --------------------------
Chat Header (Title Bar)
-------------------------- */
.chat-header {
  padding: 1.25rem;
  background-color: #2563eb;
  color: #ffffff;
  text-align: center;
}

.chat-header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}

/* --------------------------
Chat Messages Area
-------------------------- */
.chat-messages {
  flex: 1;
  padding: 1.25rem;
  background-color: #f9fafb;
  overflow-y: auto;
  scroll-behavior: smooth;
}

.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
  background-color: rgba(156, 163, 175, 0.5);
  border-radius: 3px;
}

/* --------------------------
Message Rows (User/Assistant)
-------------------------- */
.message-row {
  display: flex;
  margin-bottom: 1rem;
  max-width: 100%;
  animation: fadeIn 0.2s ease-out;
}

.message-row--user {
  justify-content: flex-end;
}

.message-row--assistant {
  justify-content: flex-start;
}

/* --------------------------
Message Bubbles (Content Boxes)
-------------------------- */
.message-bubble {
  padding: 0.875rem 1.25rem;
  border-radius: 1.25rem;
  max-width: 70%;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s ease-out;
  line-height: 1.6;
}

.message-bubble.isNew {
  transform: scale(1.05);
}

.message-bubble--user {
  background-color: #2563eb;
  color: #ffffff;
  border-bottom-right-radius: 0.375rem;
}

/* Ensure markdown elements are visible in user messages */
.message-bubble--user strong {
  color: #ffffff;
}

.message-bubble--assistant {
  background-color: #e5e7eb;
  color: #111827;
  border-bottom-left-radius: 0.375rem;
}

/* Markdown Styling */
.message-bubble strong {
  font-weight: 600;
}

.message-bubble em {
  font-style: italic;
}

.message-bubble ul,
.message-bubble ol {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.message-bubble li {
  margin-bottom: 0.25rem;
}

.message-bubble p {
  margin: 0.5rem 0;
}

/* --------------------------
Input Container
-------------------------- */
.input-container {
  padding: 1rem;
  border-top: 1px solid #e5e7eb;
  display: flex;
  gap: 0.75rem;
  background-color: #ffffff;
}

.input-field {
  flex: 1;
  border: 1px solid #d1d5db;
  border-radius: 0.75rem;
  padding: 0.875rem 1.25rem;
  resize: none;
  min-height: 48px;
  max-height: 140px;
  font-size: 1rem;
  transition: all 0.2s ease;
}

.input-field:focus {
  outline: none;
  border-color: #2563eb;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}

.send-btn {
  padding: 0.875rem 1.75rem;
  background-color: #2563eb;
  color: #ffffff;
  border: none;
  border-radius: 0.75rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease;
  white-space: nowrap;
}

.send-btn:hover:not(:disabled) {
  background-color: #1d4ed8;
}

.send-btn:disabled {
  background-color: #93c5fd;
  cursor: not-allowed;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
