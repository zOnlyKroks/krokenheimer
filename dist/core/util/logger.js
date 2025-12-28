export class Logger {
    formatMessage(level, message, ...args) {
        const timestamp = new Date().toISOString();
        return `[${timestamp}] [${level}] ${message}`;
    }
    info(message, ...args) {
        console.log(this.formatMessage("INFO", message), ...args);
    }
    error(message, ...args) {
        console.error(this.formatMessage("ERROR", message), ...args);
    }
    warn(message, ...args) {
        console.warn(this.formatMessage("WARN", message), ...args);
    }
    debug(message, ...args) {
        console.debug(this.formatMessage("DEBUG", message), ...args);
    }
}
// Export a default logger instance
export const logger = new Logger();
//# sourceMappingURL=logger.js.map