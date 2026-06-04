import { FormEvent, useEffect, useMemo, useState } from "react";

import { createThread, listMessages, listThreads, postMessage } from "./api";
import type { MessageOut, ThreadSummary } from "./types";

export function App(): JSX.Element {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string>("");
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap(): Promise<void> {
    setError("");
    try {
      const threadResp = await listThreads();
      let foundThreads = threadResp.threads;
      let selectedId = foundThreads[0]?.thread_id || "";

      if (!selectedId) {
        const created = await createThread();
        selectedId = created.thread_id;
        foundThreads = [{ thread_id: selectedId }];
      }

      setThreads(foundThreads);
      setActiveThreadId(selectedId);
      const history = await listMessages(selectedId);
      setMessages(history.messages);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to initialize chat.");
    }
  }

  async function refreshMessages(threadId: string): Promise<void> {
    const history = await listMessages(threadId);
    setMessages(history.messages);
  }

  async function handleNewThread(): Promise<void> {
    setError("");
    setIsLoading(true);
    try {
      const created = await createThread();
      const nextThreads = [
        { thread_id: created.thread_id },
        ...threads,
      ];
      setThreads(nextThreads);
      setActiveThreadId(created.thread_id);
      setMessages([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create thread.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleSelectThread(threadId: string): Promise<void> {
    if (threadId === activeThreadId) {
      return;
    }
    setError("");
    setIsLoading(true);
    try {
      setActiveThreadId(threadId);
      await refreshMessages(threadId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load thread.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const content = input.trim();
    if (!content || !activeThreadId || isLoading) {
      return;
    }

    setError("");
    setIsLoading(true);

    const optimisticMessage: MessageOut = {
      id: -Date.now(),
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, optimisticMessage]);
    setInput("");

    try {
      const response = await postMessage(activeThreadId, content);
      const assistantMessage: MessageOut = {
        id: Date.now(),
        role: "assistant",
        content: response.response,
        created_at: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      await refreshMessages(activeThreadId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message.");
      await refreshMessages(activeThreadId);
    } finally {
      setIsLoading(false);
    }
  }

  const placeholder = useMemo(() => {
    if (!activeThreadId) {
      return "Preparing chat...";
    }
    return "Ask anything about the indexed papers...";
  }, [activeThreadId]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Researcher Chat</h1>
          <button type="button" className="secondary" onClick={() => void handleNewThread()} disabled={isLoading}>
            New
          </button>
        </div>
        <div className="thread-list">
          {threads.map((thread) => (
            <button
              key={thread.thread_id}
              type="button"
              className={thread.thread_id === activeThreadId ? "thread-item active" : "thread-item"}
              onClick={() => void handleSelectThread(thread.thread_id)}
              disabled={isLoading}
            >
              {thread.thread_id.slice(0, 8)}
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-panel">
        <div className="messages">
          {messages.length === 0 ? (
            <p className="empty">No messages yet. Ask your first question.</p>
          ) : (
            messages.map((message) => (
              <article key={`${message.id}-${message.created_at}`} className={`message ${message.role}`}>
                <header>{message.role === "user" ? "You" : "Model"}</header>
                <p>{message.content}</p>
              </article>
            ))
          )}
        </div>

        <form className="composer" onSubmit={(event) => void handleSubmit(event)}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={3}
            placeholder={placeholder}
            disabled={isLoading || !activeThreadId}
          />
          <button type="submit" disabled={isLoading || !activeThreadId || !input.trim()}>
            {isLoading ? "Sending..." : "Send"}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}
      </main>
    </div>
  );
}
