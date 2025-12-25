export interface StoredMessage {
  id: string;
  channelId: string;
  channelName: string;
  authorId: string;
  authorName: string;
  content: string;
  timestamp: number;
  hasAttachments: boolean;
  replyToId?: string;
}

export interface LLMConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  contextWindow: number;
}

export interface MessageGenerationConfig {
  enabled: boolean;
  minIntervalMinutes: number;
  maxIntervalMinutes: number;
  channelIds?: string[];
  excludeChannelIds?: string[];
}

export interface RAGContext {
  messages: StoredMessage[];
  channelContext: string;
  recentMessagesCount: number;
}

export interface GeneratedMessage {
  content: string;
  channelId: string;
  confidence: number;
}