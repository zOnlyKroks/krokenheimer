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
    // Get relevant historical messages using vector search
    let historicalContext: StoredMessage[] = [];
    if (context.length > 0) {
      const vectorStoreService = (await import('./VectorStoreService.js')).default;
      const recentTopics = context.slice(-5).map(m => m.content).join(' ');
      historicalContext = await vectorStoreService.findSimilarMessages(recentTopics, undefined, 20);

      const recentIds = new Set(context.map(m => m.id));
      historicalContext = historicalContext.filter(m => !recentIds.has(m.id));
    }

    // Build prompt
    const prompt = this.buildContextPrompt(context, channelName, historicalContext);

    // Generate using trained model
    return await this.generate(prompt);
  }

  async generateMentionResponse(context: StoredMessage[], messageContent: string, authorName: string): Promise<string> {
    const prompt = this.buildMentionPrompt(context, messageContent, authorName);
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
      prompt += '\n\nRelevant past conversations:\n';
      historicalContext.slice(0, 10).forEach(msg => {
        prompt += `${msg.authorName}: ${msg.content}\n`;
      });
    }

    prompt += `\n\nCurrent conversation in #${channelName}:\n`;
    context.forEach(msg => {
      prompt += `${msg.authorName}: ${msg.content}\n`;
    });

    prompt += '\n<|assistant|>\n';
    return prompt;
  }

  private buildMentionPrompt(context: StoredMessage[], messageContent: string, authorName: string): string {
    let prompt = '<|system|>\nYou are a Discord bot. Someone mentioned you. Respond naturally.';

    prompt += '\n\n<|user|>\n';
    if (context.length > 0) {
      prompt += 'Recent conversation:\n';
      context.forEach(msg => {
        prompt += `${msg.authorName}: ${msg.content}\n`;
      });
    }
    prompt += `\n${authorName}: ${messageContent}`;

    prompt += '\n\n<|assistant|>\n';
    return prompt;
  }

  private cleanResponse(text: string): string {
    // Remove any remaining special tokens
    text = text.replace(/<\|[^|]+\|>/g, '');

    // Remove markdown
    text = text.replace(/```[\s\S]*?```/g, '');
    text = text.replace(/`[^`]+`/g, '');

    // Remove AI prefixes
    text = text.replace(/^(Here's a message|Here is a message|Message:|Response:)\s*/i, '');

    // Take first line
    const lines = text.split('\n').filter(l => l.trim().length > 0);
    text = lines[0] || text;

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
