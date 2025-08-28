"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GifPlugin = void 0;
var canvas_1 = require("canvas");
var fs_1 = require("fs");
var path = require("path");
var gifencoder_1 = require("gifencoder");
var node_fetch_1 = require("node-fetch");
var url_1 = require("url");
var GifPlugin = /** @class */ (function () {
    function GifPlugin() {
        this.name = "GifPlugin";
        this.description = "Create GIFs from images";
        this.version = "1.0.0";
        this.commands = [
            {
                name: "gif",
                description: "Create a GIF from attached images",
                usage: "!gif (attach images)",
                cooldown: 10,
                execute: this.createGif.bind(this)
            },
            {
                name: "gifhelp",
                description: "Show help for GIF commands",
                aliases: ["gh"],
                execute: this.showHelp.bind(this)
            }
        ];
    }
    GifPlugin.prototype.createGif = function (message, args) {
        return __awaiter(this, void 0, void 0, function () {
            var urls, options, gifPath, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!(message.attachments.size <= 0)) return [3 /*break*/, 2];
                        return [4 /*yield*/, message.reply("❌ Please attach images to create a GIF")];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                    case 2:
                        _a.trys.push([2, 6, , 8]);
                        return [4 /*yield*/, message.reply("🔄 Creating GIF...")];
                    case 3:
                        _a.sent();
                        urls = __spreadArray([], message.attachments.values(), true).map(function (att) { return att.url; });
                        options = this.parseGifOptions(args);
                        return [4 /*yield*/, this.createGifFromUrls(urls, options)];
                    case 4:
                        gifPath = _a.sent();
                        return [4 /*yield*/, message.reply({
                                content: "Here's your GIF 🎞️",
                                files: [gifPath],
                            })];
                    case 5:
                        _a.sent();
                        (0, fs_1.unlinkSync)(gifPath);
                        return [3 /*break*/, 8];
                    case 6:
                        error_1 = _a.sent();
                        console.error(error_1);
                        return [4 /*yield*/, message.reply("❌ Failed to create GIF")];
                    case 7:
                        _a.sent();
                        return [3 /*break*/, 8];
                    case 8: return [2 /*return*/];
                }
            });
        });
    };
    GifPlugin.prototype.showHelp = function (message) {
        return __awaiter(this, void 0, void 0, function () {
            var helpText;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        helpText = "\n**\uD83C\uDF9E\uFE0F GIF Plugin Help**\n\n**Commands:**\n`!gif` - Create a GIF from attached images\n`!gifhelp` - Show this help message\n\n**Options:**\n`--delay=500` - Set frame delay in milliseconds (default: 500)\n`--quality=10` - Set quality 1-20 (lower = better, default: 10)\n`--repeat=0` - Set repeat count (0 = infinite, default: 0)\n`--width=300` - Resize width (maintains aspect ratio)\n`--height=300` - Resize height (maintains aspect ratio)\n\n**Example:**\n`!gif --delay=200 --quality=5` (attach images)\n    ";
                        return [4 /*yield*/, message.reply(helpText)];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    GifPlugin.prototype.parseGifOptions = function (args) {
        var _a, _b, _c, _d, _e;
        var options = {
            delay: 500,
            quality: 10,
            repeat: 0
        };
        for (var _i = 0, args_1 = args; _i < args_1.length; _i++) {
            var arg = args_1[_i];
            if (arg.startsWith("--delay=")) {
                options.delay = parseInt((_a = arg.split("=")[1]) !== null && _a !== void 0 ? _a : "500") || 500;
            }
            else if (arg.startsWith("--quality=")) {
                options.quality = Math.max(1, Math.min(20, parseInt((_b = arg.split("=")[1]) !== null && _b !== void 0 ? _b : "10") || 10));
            }
            else if (arg.startsWith("--repeat=")) {
                options.repeat = parseInt((_c = arg.split("=")[1]) !== null && _c !== void 0 ? _c : "0") || 0;
            }
            else if (arg.startsWith("--width=")) {
                options.width = parseInt((_d = arg.split("=")[1]) !== null && _d !== void 0 ? _d : "0");
            }
            else if (arg.startsWith("--height=")) {
                options.height = parseInt((_e = arg.split("=")[1]) !== null && _e !== void 0 ? _e : "0");
            }
        }
        return options;
    };
    GifPlugin.prototype.createGifFromUrls = function (urls, options) {
        return __awaiter(this, void 0, void 0, function () {
            var images, _a, width, height, encoder, __filename, __dirname, gifPath, stream, canvas, ctx, _i, images_1, img;
            var _this = this;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0: return [4 /*yield*/, Promise.all(urls.map(function (url) { return _this.loadImageFromUrl(url); }))];
                    case 1:
                        images = _b.sent();
                        if (images.length === 0) {
                            throw new Error("No images loaded");
                        }
                        _a = this.calculateDimensions(images[0], options), width = _a.width, height = _a.height;
                        encoder = new gifencoder_1.default(width, height);
                        __filename = (0, url_1.fileURLToPath)(import.meta.url);
                        __dirname = path.dirname(__filename);
                        gifPath = path.join(__dirname, "output_".concat(Date.now(), ".gif"));
                        stream = (0, fs_1.createWriteStream)(gifPath);
                        encoder.createReadStream().pipe(stream);
                        encoder.start();
                        encoder.setRepeat(options.repeat);
                        encoder.setDelay(options.delay);
                        encoder.setQuality(options.quality);
                        canvas = (0, canvas_1.createCanvas)(width, height);
                        ctx = canvas.getContext("2d");
                        for (_i = 0, images_1 = images; _i < images_1.length; _i++) {
                            img = images_1[_i];
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(img, 0, 0, width, height);
                            encoder.addFrame(ctx);
                        }
                        encoder.finish();
                        return [4 /*yield*/, new Promise(function (resolve) {
                                stream.on("finish", function () { return resolve(); });
                            })];
                    case 2:
                        _b.sent();
                        return [2 /*return*/, gifPath];
                }
            });
        });
    };
    GifPlugin.prototype.calculateDimensions = function (firstImage, options) {
        var width = firstImage.width;
        var height = firstImage.height;
        if (options.width && options.height) {
            width = options.width;
            height = options.height;
        }
        else if (options.width) {
            width = options.width;
            height = Math.round((firstImage.height / firstImage.width) * options.width);
        }
        else if (options.height) {
            height = options.height;
            width = Math.round((firstImage.width / firstImage.height) * options.height);
        }
        return { width: width, height: height };
    };
    GifPlugin.prototype.loadImageFromUrl = function (url) {
        return __awaiter(this, void 0, void 0, function () {
            var res, buffer;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, (0, node_fetch_1.default)(url)];
                    case 1:
                        res = _a.sent();
                        if (!res.ok)
                            throw new Error("Failed to fetch ".concat(url));
                        return [4 /*yield*/, res.arrayBuffer()];
                    case 2:
                        buffer = _a.sent();
                        return [2 /*return*/, (0, canvas_1.loadImage)(Buffer.from(buffer))];
                }
            });
        });
    };
    return GifPlugin;
}());
exports.GifPlugin = GifPlugin;
