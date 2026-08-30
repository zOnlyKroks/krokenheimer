import Database from 'better-sqlite3';
import { join } from 'path';
import { mkdirSync, existsSync } from 'fs';

export interface StoredMessage {
  id: number;
  messageId: string;
  guildId: string;
  channelId: string;
  userId: string;
  username: string;
  content: string;
  timestamp: number;
}

export class MessageDatabase {
  private db: Database.Database;

  constructor(dbPath: string = './data/messages.db') {
    const dir = join(dbPath, '..');
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    this.db = new Database(dbPath);
    this.initialize();
  }

  private initialize(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        messageId TEXT NOT NULL,
        guildId TEXT NOT NULL,
        channelId TEXT NOT NULL,
        userId TEXT NOT NULL,
        username TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_guild ON messages(guildId);
      CREATE INDEX IF NOT EXISTS idx_channel ON messages(channelId);
      CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp);
    `);
  }

  public storeMessage(
    messageId: string,
    guildId: string,
    channelId: string,
    userId: string,
    username: string,
    content: string,
    timestamp: number
  ): void {
    const stmt = this.db.prepare(`
      INSERT INTO messages (messageId, guildId, channelId, userId, username, content, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(messageId, guildId, channelId, userId, username, content, timestamp);
  }

  public getRecentMessages(guildId: string, limit: number = 50): StoredMessage[] {
    const stmt = this.db.prepare(`
      SELECT * FROM messages
      WHERE guildId = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    return stmt.all(guildId, limit) as StoredMessage[];
  }

  public getChannelMessages(channelId: string, limit: number = 30): StoredMessage[] {
    const stmt = this.db.prepare(`
      SELECT * FROM messages
      WHERE channelId = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    return stmt.all(channelId, limit) as StoredMessage[];
  }

  public getMessageCount(guildId: string): number {
    const stmt = this.db.prepare(`
      SELECT COUNT(*) as count FROM messages WHERE guildId = ?
    `);

    const result = stmt.get(guildId) as { count: number };
    return result.count;
  }

  public close(): void {
    this.db.close();
  }
}