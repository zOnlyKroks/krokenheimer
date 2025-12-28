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
    // Try multiple strategies before giving up
    const strategies = [
      { temp: '0.9', max_length: '100' },
      { temp: '0.7', max_length: '80' },
      { temp: '0.5', max_length: '120' }
    ];

    for (const strategy of strategies) {
      try {
        const result = await this.tryGenerate(prompt, strategy.temp, strategy.max_length);
        if (result && result.length > 0) {
          return result;
        }
      } catch (error) {
        console.warn(`Generation strategy failed (temp: ${strategy.temp}):`, error);
        // Continue to next strategy
      }
    }

    // If all strategies fail, return a simple acknowledgment
    return "👍";
  }

  private async tryGenerate(prompt: string, temperature: string, maxLength: string): Promise<string> {
    return new Promise(async (resolve, reject) => {
      const pythonCmd = await fs.access('./venv/bin/python3')
        .then(() => './venv/bin/python3')
        .catch(() => 'python3');

      const pythonProcess = spawn(pythonCmd, [
        './scripts/generate.py',
        this.modelPath,
        '--prompt', prompt,
        '--max_length', maxLength,
        '--temperature', temperature
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
          reject(new Error(`Generation failed with code ${code}: ${stderr}`));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(error);
      });
    });
  }

  private buildContextPrompt(context: StoredMessage[], channelName: string, historicalContext: StoredMessage[]): string {
    let prompt = '';

    if (historicalContext.length > 0) {
      historicalContext.slice(0, 3).forEach(msg => {
        const content = msg.content.length > 100 ? msg.content.substring(0, 100) + '...' : msg.content;
        prompt += `${msg.authorName}: ${content}\n`;
      });
      prompt += '\n';
    }

    context.forEach(msg => {
      const content = msg.content.length > 150 ? msg.content.substring(0, 150) + '...' : msg.content;
      prompt += `${msg.authorName}: ${content}\n`;
    });

    prompt += 'Krokenheimer: ';
    return prompt;
  }

  private buildMentionPrompt(context: StoredMessage[], messageContent: string, authorName: string): string {
    let prompt = '';

    if (context.length > 0) {
      context.forEach(msg => {
        const content = msg.content.length > 150 ? msg.content.substring(0, 150) + '...' : msg.content;
        prompt += `${msg.authorName}: ${content}\n`;
      });
    }

    const truncatedContent = messageContent.length > 200 ? messageContent.substring(0, 200) + '...' : messageContent;
    prompt += `${authorName}: ${truncatedContent}\n`;
    prompt += 'Krokenheimer: ';

    return prompt;
  }

  private cleanResponse(text: string): string {
    // Remove special tokens
    text = text.replace(/<\|endoftext\|>/gi, '');
    text = text.replace(/<\|assistant\|>/gi, '');
    text = text.replace(/<\|system\|>/gi, '');
    text = text.replace(/<\|user\|>/gi, '');
    text = text.replace(/<\|pad\|>/gi, '');

    // Remove any remaining bot name prefix
    text = text.replace(/^Krokenheimer:\s*/i, '');

    // Clean up whitespace
    text = text.trim();

    // If multi-line, take first meaningful line
    const lines = text.split('\n').filter(l => l.trim().length > 2);
    if (lines.length > 0) {
      // @ts-ignore
        text = lines[0].trim();
    }

    // Limit length to keep it conversational
    if (text.length > 300) {
      text = text.substring(0, 297) + '...';
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
