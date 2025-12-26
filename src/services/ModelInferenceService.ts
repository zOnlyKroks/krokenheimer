import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import { StoredMessage } from '../types/llm.js';

/**
 * Model Inference Service
 * Uses YOUR trained-from-scratch GPT-2 model for generating responses
 * NO Ollama, NO pre-trained models - pure Discord learning
 */
export class ModelInferenceService {
  private modelPath = './data/models/krokenheimer';
  private isModelLoaded = false;

  async checkModelExists(): Promise<boolean> {
    try {
      await fs.access(`${this.modelPath}/pytorch_model.bin`);
      await fs.access(`${this.modelPath}/config.json`);
      await fs.access(`${this.modelPath}/tokenizer.json`);
      return true;
    } catch {
      return false;
    }
  }

  async generateMessage(context: StoredMessage[], channelName: string, channelId?: string): Promise<string> {
    // Get relevant historical messages using vector search (limited to save context window)
    let historicalContext: StoredMessage[] = [];
    if (context.length > 0) {
      const vectorStoreService = (await import('./VectorStoreService.js')).default;
      const recentTopics = context.slice(-3).map(m => m.content).join(' ');
      historicalContext = await vectorStoreService.findSimilarMessages(recentTopics, undefined, 5);

      const recentIds = new Set(context.map(m => m.id));
      historicalContext = historicalContext.filter(m => !recentIds.has(m.id));
    }

    // Limit recent context to last 10 messages to stay within 512 token limit
    const limitedContext = context.slice(-10);

    // Build prompt
    const prompt = this.buildContextPrompt(limitedContext, channelName, historicalContext);

    // Generate using trained model
    return await this.generate(prompt);
  }

  async generateMentionResponse(context: StoredMessage[], messageContent: string, authorName: string): Promise<string> {
    // Limit context for mentions too
    const limitedContext = context.slice(-8);
    const prompt = this.buildMentionPrompt(limitedContext, messageContent, authorName);
    return await this.generate(prompt);
  }

  private async generate(prompt: string): Promise<string> {
    return new Promise(async (resolve, reject) => {
      const pythonCmd = await fs.access('./venv/bin/python3')
        .then(() => './venv/bin/python3')
        .catch(() => 'python3');

      const pythonProcess = spawn(pythonCmd, [
        './scripts/generate.py',
        this.modelPath,
        '--prompt', prompt,
        '--max_length', '100',
        '--temperature', '0.9'
      ]);

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          const generated = stdout.trim();
          const cleaned = this.cleanResponse(generated);
          resolve(cleaned);
        } else {
          console.error('Generation failed:', stderr);
          reject(new Error(`Generation failed with code ${code}`));
        }
      });

      pythonProcess.on('error', (error) => {
        console.error('Failed to spawn Python process:', error);
        reject(error);
      });
    });
  }

  private buildContextPrompt(context: StoredMessage[], channelName: string, historicalContext: StoredMessage[]): string {
    let prompt = '<|system|>\nYou are a member of this Discord server. Match the tone and style of the conversations.';

    if (historicalContext.length > 0) {
      prompt += '\n\nRelevant past:\n';
      historicalContext.slice(0, 5).forEach(msg => {
        // Truncate long messages to save tokens
        const content = msg.content.length > 100 ? msg.content.substring(0, 100) + '...' : msg.content;
        prompt += `${msg.authorName}: ${content}\n`;
      });
    }

    prompt += `\n\nCurrent in #${channelName}:\n`;
    context.forEach(msg => {
      // Truncate long messages to save tokens
      const content = msg.content.length > 150 ? msg.content.substring(0, 150) + '...' : msg.content;
      prompt += `${msg.authorName}: ${content}\n`;
    });

    prompt += '\n<|assistant|>\n';
    return prompt;
  }

  private buildMentionPrompt(context: StoredMessage[], messageContent: string, authorName: string): string {
    let prompt = '<|system|>\nYou are a Discord bot. Someone mentioned you. Respond naturally.';

    prompt += '\n\n<|user|>\n';
    if (context.length > 0) {
      prompt += 'Recent:\n';
      context.forEach(msg => {
        // Truncate long messages to save tokens
        const content = msg.content.length > 150 ? msg.content.substring(0, 150) + '...' : msg.content;
        prompt += `${msg.authorName}: ${content}\n`;
      });
    }

    // Truncate the mention message too if needed
    const truncatedContent = messageContent.length > 200 ? messageContent.substring(0, 200) + '...' : messageContent;
    prompt += `\n${authorName}: ${truncatedContent}`;

    prompt += '\n\n<|assistant|>\n';
    return prompt;
  }

  private cleanResponse(text: string): string {
    // Aggressive special token removal (all possible variations)
    text = text.replace(/<\|assistant\|>/gi, '');
    text = text.replace(/<\|system\|>/gi, '');
    text = text.replace(/<\|user\|>/gi, '');
    text = text.replace(/<\|pad\|>/gi, '');
    text = text.replace(/<\|endoftext\|>/gi, '');

    // Remove partial tokens (missing opening <)
    text = text.replace(/\|assistant\|>/gi, '');
    text = text.replace(/\|system\|>/gi, '');
    text = text.replace(/\|user\|>/gi, '');
    text = text.replace(/\|pad\|>/gi, '');
    text = text.replace(/\|endoftext\|>/gi, '');

    // Remove tokens missing closing >
    text = text.replace(/<\|assistant\|/gi, '');
    text = text.replace(/<\|system\|/gi, '');
    text = text.replace(/<\|user\|/gi, '');
    text = text.replace(/<\|pad\|/gi, '');
    text = text.replace(/<\|endoftext\|/gi, '');

    // Catch any remaining angle bracket patterns
    text = text.replace(/<\|[^>|]+\|>/g, '');
    text = text.replace(/\|[^>|]+\|>/g, '');

    // Remove markdown code blocks
    text = text.replace(/```[\s\S]*?```/g, '');
    text = text.replace(/`[^`]+`/g, '');

    // Remove AI prefixes
    text = text.replace(/^(Here's a message|Here is a message|Message:|Response:|Assistant:)\s*/i, '');

    // Remove leading/trailing whitespace and newlines
    text = text.trim();

    // Take first non-empty line if multi-line
    const lines = text.split('\n').filter(l => l.trim().length > 0);
    if (lines.length > 0) {
      text = lines[0];
    }

    // Final pass: remove any remaining fragments
    text = text.replace(/[<|]+(assistant|system|user|pad|endoftext)[|>]+/gi, '');

    // Limit length
    if (text.length > 500) {
      text = text.substring(0, 497) + '...';
    }

    return text.trim();
  }

  getConfig() {
    return {
      model: 'krokenheimer (trained from scratch)',
      temperature: 0.9,
      maxTokens: 100,
      contextWindow: 512
    };
  }
}

const modelInferenceService = new ModelInferenceService();
export default modelInferenceService;
