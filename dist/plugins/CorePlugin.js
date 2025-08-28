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
Object.defineProperty(exports, "__esModule", { value: true });
exports.CorePlugin = void 0;
var CorePlugin = /** @class */ (function () {
    function CorePlugin() {
        this.name = "CorePlugin";
        this.description = "Core bot functionality";
        this.version = "1.0.0";
        this.bot = null;
        this.commands = [
            {
                name: "help",
                description: "Show available commands",
                aliases: ["h", "commands"],
                execute: this.showHelp.bind(this),
            },
            {
                name: "ping",
                description: "Check bot latency",
                execute: this.ping.bind(this),
            },
            {
                name: "plugins",
                description: "List loaded plugins",
                execute: this.listPlugins.bind(this),
            },
        ];
    }
    CorePlugin.prototype.initialize = function (client, bot) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                this.bot = bot;
                console.log("Core plugin initialized");
                return [2 /*return*/];
            });
        });
    };
    CorePlugin.prototype.showHelp = function (message, args) {
        return __awaiter(this, void 0, void 0, function () {
            var commandName_1, commands_1, command, helpText_1, commands, commandList, helpText;
            var _a, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        if (!!this.bot) return [3 /*break*/, 2];
                        return [4 /*yield*/, message.reply("❌ Bot reference not available")];
                    case 1:
                        _c.sent();
                        return [2 /*return*/];
                    case 2:
                        if (!(args.length > 0)) return [3 /*break*/, 6];
                        commandName_1 = (_b = (_a = args[0]) === null || _a === void 0 ? void 0 : _a.toLowerCase()) !== null && _b !== void 0 ? _b : "noCommand";
                        commands_1 = this.bot.getCommands();
                        command = commands_1.find(function (cmd) { var _a; return cmd.name === commandName_1 || ((_a = cmd.aliases) === null || _a === void 0 ? void 0 : _a.includes(commandName_1)); });
                        if (!command) return [3 /*break*/, 4];
                        helpText_1 = "\n**\uD83D\uDCD6 Command: ".concat(command.name, "**\n").concat(command.description, "\n\n**Usage:** `!").concat(command.usage || command.name, "`\n").concat(command.aliases ? "**Aliases:** ".concat(command.aliases.join(", ")) : "", "\n").concat(command.cooldown ? "**Cooldown:** ".concat(command.cooldown, "s") : "", "\n        ");
                        return [4 /*yield*/, message.reply(helpText_1)];
                    case 3:
                        _c.sent();
                        return [2 /*return*/];
                    case 4: return [4 /*yield*/, message.reply("\u274C Command `".concat(commandName_1, "` not found."))];
                    case 5:
                        _c.sent();
                        return [2 /*return*/];
                    case 6:
                        commands = this.bot.getCommands();
                        commandList = commands
                            .map(function (cmd) { return "\u2022 `!".concat(cmd.name, "` - ").concat(cmd.description); })
                            .join("\n");
                        helpText = "\n**\uD83E\uDD16 Bot Commands**\n\nUse `!help <command>` for detailed info about a specific command.\n\n**Available Commands:**\n".concat(commandList, "\n    ");
                        return [4 /*yield*/, message.reply(helpText)];
                    case 7:
                        _c.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    CorePlugin.prototype.ping = function (message) {
        return __awaiter(this, void 0, void 0, function () {
            var sent, latency, wsLatency;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, message.reply("🏓 Pinging...")];
                    case 1:
                        sent = _a.sent();
                        latency = sent.createdTimestamp - message.createdTimestamp;
                        wsLatency = message.client.ws.ping;
                        return [4 /*yield*/, sent.edit("\uD83C\uDFD3 Pong!\n**Message Latency:** ".concat(latency, "ms\n**WebSocket Latency:** ").concat(wsLatency, "ms"))];
                    case 2:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    CorePlugin.prototype.listPlugins = function (message) {
        return __awaiter(this, void 0, void 0, function () {
            var plugins, pluginList;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!!this.bot) return [3 /*break*/, 2];
                        return [4 /*yield*/, message.reply("❌ Bot reference not available")];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                    case 2:
                        plugins = this.bot.getLoadedPlugins();
                        pluginList = plugins
                            .map(function (name) { return "\u2022 ".concat(name); })
                            .join("\n");
                        return [4 /*yield*/, message.reply("**\uD83D\uDCE6 Loaded Plugins (".concat(plugins.length, ")**\n").concat(pluginList))];
                    case 3:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    return CorePlugin;
}());
exports.CorePlugin = CorePlugin;
