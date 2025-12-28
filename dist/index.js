import dotenv from "dotenv";
import { existsSync } from "fs";
import { ExtensibleBot } from "./core/Bot.js";
import { GifPlugin } from "./plugins/GifPlugin.js";
import { CorePlugin } from "./plugins/CorePlugin.js";
import { BioinformaticsPlugin } from "./plugins/BioinformaticsPlugin.js";
import { CalculatorPlugin } from "./plugins/CalculatorPlugin.js";
import { ASCIIArtPlugin } from "./plugins/ASCIIArtPlugin.js";
import { DownCheckerPlugin } from "./plugins/DownCheckerPlugin.js";
import { LLMPlugin } from "./plugins/LLMPlugin.js";
import { RemoteTrainingApiService } from "./services/RemoteTrainingApiService.js";
import messageStorageService from "./services/MessageStorageService.js";
import fineTuningService from "./services/FineTuningService.js";
// Only load .env file if it exists (for local development)
// In Docker, environment variables are passed directly
if (existsSync('.env')) {
    dotenv.config();
}
else {
    console.log('No .env file found, using environment variables from system');
}
async function main() {
    // Validate required environment variables
    if (!process.env.BOT_TOKEN) {
        console.error('❌ BOT_TOKEN environment variable is not set!');
        console.error('   Set it in .env file or pass it via docker-compose');
        process.exit(1);
    }
    console.log('✅ BOT_TOKEN is set');
    console.log(`✅ BOT_OWNERS: ${process.env.BOT_OWNERS || '(none)'}`);
    const config = {
        prefix: "!",
        token: process.env.BOT_TOKEN,
        owners: process.env.BOT_OWNERS?.split(",") || [],
        maxFileSize: 80 * 1024 * 1024, // 80MB
        allowedFileTypes: [".png", ".jpg", ".jpeg", ".gif", ".webp"],
    };
    const bot = new ExtensibleBot(config);
    // Initialize Remote Training API Service
    const remoteApiService = new RemoteTrainingApiService(messageStorageService, fineTuningService);
    try {
        await bot.loadPlugin(new CorePlugin());
        await bot.loadPlugin(new GifPlugin());
        await bot.loadPlugin(new BioinformaticsPlugin());
        await bot.loadPlugin(new CalculatorPlugin());
        await bot.loadPlugin(new ASCIIArtPlugin());
        await bot.loadPlugin(new DownCheckerPlugin());
        await bot.loadPlugin(new LLMPlugin());
        await bot.start();
        // Start Remote Training API if enabled
        const enableRemoteApi = process.env.REMOTE_API_ENABLED !== 'false';
        if (enableRemoteApi) {
            try {
                await remoteApiService.start();
                console.log("🌐 Remote Training API started successfully!");
            }
            catch (error) {
                console.error("❌ Failed to start Remote Training API:", error);
                console.log("ℹ️ Bot will continue running without remote API");
            }
        }
        else {
            console.log("🔌 Remote Training API disabled via REMOTE_API_ENABLED=false");
        }
        console.log("🚀 Bot started successfully!");
        // Graceful shutdown
        process.on('SIGINT', async () => {
            console.log('\n🛑 Shutting down gracefully...');
            try {
                if (remoteApiService.isRunning()) {
                    await remoteApiService.stop();
                }
                process.exit(0);
            }
            catch (error) {
                console.error('Error during shutdown:', error);
                process.exit(1);
            }
        });
    }
    catch (error) {
        console.error("Failed to start bot:", error);
        process.exit(1);
    }
}
main().catch(console.error);
//# sourceMappingURL=index.js.map