import type {
  CreateThreadResponse,
  ListMessagesResponse,
  ListThreadsResponse,
  PostMessageResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() || "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Request failed (${response.status}): ${body || response.statusText}`);
  }

  return (await response.json()) as T;
}

export function createThread(): Promise<CreateThreadResponse> {
  return request<CreateThreadResponse>("/threads", { method: "POST" });
}

export function listThreads(): Promise<ListThreadsResponse> {
  return request<ListThreadsResponse>("/threads");
}

export function listMessages(threadId: string): Promise<ListMessagesResponse> {
  return request<ListMessagesResponse>(`/threads/${threadId}/messages`);
}

export function postMessage(threadId: string, content: string): Promise<PostMessageResponse> {
  return request<PostMessageResponse>(`/threads/${threadId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}
