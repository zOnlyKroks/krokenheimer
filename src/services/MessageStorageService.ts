import Database from 'better-sqlite3';
import { StoredMessage } from '../types/llm.js';
import { Message } from 'discord.js';

export class MessageStorageService {
  private db: Database.Database;

  constructor(dbPath: string = './data/messages.db') {
    this.db = new Database(dbPath);
    this.initDatabase();
  }

  private initDatabase(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        channelId TEXT NOT NULL,
        channelName TEXT NOT NULL,
        authorId TEXT NOT NULL,
        authorName TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        hasAttachments INTEGER NOT NULL,
        replyToId TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_channelId ON messages(channelId);
      CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp);
      CREATE INDEX IF NOT EXISTS idx_authorId ON messages(authorId);
    `);
  }

  storeMessage(message: Message): void {
    // Skip bot messages and empty content
    if (message.author.bot || !message.content || message.content.trim() === '') {
      return;
    }

    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO messages
      (id, channelId, channelName, authorId, authorName, content, timestamp, hasAttachments, replyToId)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      message.id,
      message.channel.id,
      message.channel.isDMBased() ? 'DM' : message.channel.name || 'Unknown',
      message.author.id,
      message.author.username,
      message.content,
      message.createdTimestamp,
      message.attachments.size > 0 ? 1 : 0,
      message.reference?.messageId || null
    );
  }

  getRecentMessages(channelId: string, limit: number = 50): StoredMessage[] {
    const stmt = this.db.prepare(`
      SELECT * FROM messages
      WHERE channelId = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    const rows = stmt.all(channelId, limit) as any[];
    return rows.map(row => this.rowToMessage(row)).reverse();
  }

  getRandomMessages(count: number = 100): StoredMessage[] {
    const stmt = this.db.prepare(`
      SELECT * FROM messages
      ORDER BY RANDOM()
      LIMIT ?
    `);

    const rows = stmt.all(count) as any[];
    return rows.map(row => this.rowToMessage(row));
  }

  getMessagesByChannel(channelId: string, limit: number = 1000): StoredMessage[] {
    const stmt = this.db.prepare(`
      SELECT * FROM messages
      WHERE channelId = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    const rows = stmt.all(channelId, limit) as any[];
    return rows.map(row => this.rowToMessage(row));
  }

  getTotalMessageCount(): number {
    const stmt = this.db.prepare('SELECT COUNT(*) as count FROM messages');
    const result = stmt.get() as { count: number };
    return result.count;
  }

  getChannelMessageCount(channelId: string): number {
    const stmt = this.db.prepare('SELECT COUNT(*) as count FROM messages WHERE channelId = ?');
    const result = stmt.get(channelId) as { count: number };
    return result.count;
  }

  getActiveChannels(): Array<{ channelId: string; channelName: string; count: number }> {
    const stmt = this.db.prepare(`
      SELECT channelId, channelName, COUNT(*) as count
      FROM messages
      GROUP BY channelId, channelName
      ORDER BY count DESC
    `);

    return stmt.all() as Array<{ channelId: string; channelName: string; count: number }>;
  }

  getLastScannedTimestamp(channelId: string): number {
    const stmt = this.db.prepare(`
      SELECT MAX(timestamp) as lastTimestamp
      FROM messages
      WHERE channelId = ?
    `);

    const result = stmt.get(channelId) as { lastTimestamp: number | null };
    return result.lastTimestamp || 0;
  }

  getAllMessages(limit?: number): StoredMessage[] {
    const sql = limit
      ? `SELECT * FROM messages ORDER BY timestamp ASC LIMIT ?`
      : `SELECT * FROM messages ORDER BY timestamp ASC`;

    const stmt = this.db.prepare(sql);
    const rows = limit ? stmt.all(limit) as any[] : stmt.all() as any[];
    return rows.map(row => this.rowToMessage(row));
  }

  private rowToMessage(row: any): StoredMessage {
    return {
      id: row.id,
      channelId: row.channelId,
      channelName: row.channelName,
      authorId: row.authorId,
      authorName: row.authorName,
      content: row.content,
      timestamp: row.timestamp,
      hasAttachments: row.hasAttachments === 1,
      replyToId: row.replyToId || undefined
    };
  }

  close(): void {
    this.db.close();
  }
}

const messageStorageService = new MessageStorageService();
export default messageStorageService;