import type { BotCommand, BotPlugin } from "../types/index.js";
export declare class GifPlugin implements BotPlugin {
    name: string;
    description: string;
    version: string;
    commands: BotCommand[];
    private createGif;
    private showHelp;
    private parseGifOptions;
    private createGifFromUrls;
    private calculateDimensions;
    private loadImageFromUrl;
}
//# sourceMappingURL=GifPlugin.d.ts.map