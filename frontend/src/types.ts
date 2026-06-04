export type Role = "user" | "assistant";

export interface ThreadSummary {
  thread_id: string;
}

export interface ListThreadsResponse {
  threads: ThreadSummary[];
}

export interface CreateThreadResponse {
  thread_id: string;
}

export interface MessageOut {
  id: number;
  role: Role;
  content: string;
  created_at: string;
}

export interface ListMessagesResponse {
  thread_id: string;
  messages: MessageOut[];
}

export interface ToolCallTrace {
  tool: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  error: string | null;
}

export interface PostMessageResponse {
  thread_id: string;
  response: string;
  trace: ToolCallTrace[];
}
