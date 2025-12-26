/**
 * Training Configuration
 * Controls what messages are used for AI training
 */

export class TrainingConfig {
  private botUserId: string | null = null;
  private excludedChannelIds: Set<string> = new Set([
    '1405078879021961226' // Add channel IDs to exclude from training
  ]);

  /**
   * Set the bot's user ID (called when bot logs in)
   */
  setBotUserId(userId: string): void {
    this.botUserId = userId;
    console.log(`🤖 Bot user ID set for training filter: ${userId}`);
  }

  /**
   * Get the bot's user ID
   */
  getBotUserId(): string | null {
    return this.botUserId;
  }

  /**
   * Check if a channel should be excluded from training
   */
  isChannelExcluded(channelId: string): boolean {
    return this.excludedChannelIds.has(channelId);
  }

  /**
   * Check if a message should be included in training
   */
  shouldIncludeInTraining(authorId: string, channelId: string): boolean {
    // Exclude bot's own messages
    if (this.botUserId && authorId === this.botUserId) {
      return false;
    }

    // Exclude messages from excluded channels
    if (this.isChannelExcluded(channelId)) {
      return false;
    }

    return true;
  }

  /**
   * Add a channel to the exclusion list
   */
  addExcludedChannel(channelId: string): void {
    this.excludedChannelIds.add(channelId);
    console.log(`➕ Added channel ${channelId} to training exclusion list`);
  }

  /**
   * Remove a channel from the exclusion list
   */
  removeExcludedChannel(channelId: string): void {
    this.excludedChannelIds.delete(channelId);
    console.log(`➖ Removed channel ${channelId} from training exclusion list`);
  }

  /**
   * Get all excluded channel IDs
   */
  getExcludedChannels(): string[] {
    return Array.from(this.excludedChannelIds);
  }

  /**
   * Get training status info
   */
  getStatusInfo(): {
    botUserId: string | null;
    excludedChannels: string[];
    multiServerSupport: boolean;
  } {
    return {
      botUserId: this.botUserId,
      excludedChannels: this.getExcludedChannels(),
      multiServerSupport: true // Bot learns from ALL servers
    };
  }
}

const trainingConfig = new TrainingConfig();
export default trainingConfig;
