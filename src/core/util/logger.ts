export class Logger {
  private formatMessage(level: string, message: string, ...args: any[]): string {
    const timestamp = new Date().toISOString();
    return `[${timestamp}] [${level}] ${message}`;
  }

  info(message: string, ...args: any[]): void {
    console.log(this.formatMessage("INFO", message), ...args);
  }

  error(message: string, ...args: any[]): void {
    console.error(this.formatMessage("ERROR", message), ...args);
  }

  warn(message: string, ...args: any[]): void {
    console.warn(this.formatMessage("WARN", message), ...args);
  }

  debug(message: string, ...args: any[]): void {
    console.debug(this.formatMessage("DEBUG", message), ...args);
  }
}

// Export a default logger instance
export const logger = new Logger();