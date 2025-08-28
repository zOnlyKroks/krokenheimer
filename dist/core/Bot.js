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
exports.ExtensibleBot = void 0;
var discord_js_1 = require("discord.js");
var logger_ts_1 = require("./util/logger.ts");
var ExtensibleBot = /** @class */ (function () {
    function ExtensibleBot(config) {
        this.plugins = new Map();
        this.commands = new Map();
        this.cooldowns = new Map();
        this.config = config;
        this.logger = new logger_ts_1.Logger();
        this.client = new discord_js_1.Client({
            intents: [
                discord_js_1.GatewayIntentBits.Guilds,
                discord_js_1.GatewayIntentBits.GuildMessages,
                discord_js_1.GatewayIntentBits.MessageContent,
            ],
        });
        this.setupEventHandlers();
    }
    ExtensibleBot.prototype.setupEventHandlers = function () {
        var _this = this;
        this.client.once("ready", function () {
            var _a;
            _this.logger.info("\u2705 Bot logged in as ".concat((_a = _this.client.user) === null || _a === void 0 ? void 0 : _a.tag));
        });
        this.client.on("messageCreate", function (message) { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.handleMessage(message)];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        }); });
        process.on("SIGINT", function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.shutdown()];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        }); });
    };
    ExtensibleBot.prototype.handleMessage = function (message) {
        return __awaiter(this, void 0, void 0, function () {
            var args, commandName, command, error_1;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        if (message.author.bot || !message.content.startsWith(this.config.prefix)) {
                            return [2 /*return*/];
                        }
                        args = message.content.slice(this.config.prefix.length).trim().split(/ +/);
                        commandName = (_a = args.shift()) === null || _a === void 0 ? void 0 : _a.toLowerCase();
                        if (!commandName)
                            return [2 /*return*/];
                        command = this.commands.get(commandName) ||
                            __spreadArray([], this.commands.values(), true).find(function (cmd) { var _a; return (_a = cmd.aliases) === null || _a === void 0 ? void 0 : _a.includes(commandName); });
                        if (!command)
                            return [2 /*return*/];
                        return [4 /*yield*/, this.checkCooldown(message, command)];
                    case 1:
                        if (_b.sent())
                            return [2 /*return*/];
                        return [4 /*yield*/, this.checkPermissions(message, command)];
                    case 2:
                        if (_b.sent())
                            return [2 /*return*/];
                        _b.label = 3;
                    case 3:
                        _b.trys.push([3, 5, , 7]);
                        this.logger.info("Executing command: ".concat(command.name, " by ").concat(message.author.tag));
                        return [4 /*yield*/, command.execute(message, args, this.client)];
                    case 4:
                        _b.sent();
                        return [3 /*break*/, 7];
                    case 5:
                        error_1 = _b.sent();
                        this.logger.error("Error executing command ".concat(command.name, ":"), error_1);
                        return [4 /*yield*/, message.reply("❌ An error occurred while executing the command.")];
                    case 6:
                        _b.sent();
                        return [3 /*break*/, 7];
                    case 7: return [2 /*return*/];
                }
            });
        });
    };
    ExtensibleBot.prototype.checkCooldown = function (message, command) {
        return __awaiter(this, void 0, void 0, function () {
            var now, timestamps, cooldownAmount, expirationTime, timeLeft;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!command.cooldown || command.cooldown <= 0)
                            return [2 /*return*/, false];
                        now = Date.now();
                        timestamps = this.cooldowns.get(command.name) || new Map();
                        cooldownAmount = command.cooldown * 1000;
                        if (!timestamps.has(message.author.id)) return [3 /*break*/, 2];
                        expirationTime = timestamps.get(message.author.id) + cooldownAmount;
                        if (!(now < expirationTime)) return [3 /*break*/, 2];
                        timeLeft = (expirationTime - now) / 1000;
                        return [4 /*yield*/, message.reply("\u23F0 Please wait ".concat(timeLeft.toFixed(1), " seconds before using `").concat(command.name, "` again."))];
                    case 1:
                        _a.sent();
                        return [2 /*return*/, true];
                    case 2:
                        timestamps.set(message.author.id, now);
                        this.cooldowns.set(command.name, timestamps);
                        setTimeout(function () { return timestamps.delete(message.author.id); }, cooldownAmount);
                        return [2 /*return*/, false];
                }
            });
        });
    };
    ExtensibleBot.prototype.checkPermissions = function (message, command) {
        return __awaiter(this, void 0, void 0, function () {
            var member, hasPermission;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!command.permissions || command.permissions.length === 0)
                            return [2 /*return*/, false];
                        member = message.member;
                        if (!member)
                            return [2 /*return*/, false];
                        hasPermission = command.permissions.every(function (permission) {
                            return member.permissions.has(permission);
                        });
                        if (!!hasPermission) return [3 /*break*/, 2];
                        return [4 /*yield*/, message.reply("❌ You don't have permission to use this command.")];
                    case 1:
                        _a.sent();
                        return [2 /*return*/, true];
                    case 2: return [2 /*return*/, false];
                }
            });
        });
    };
    ExtensibleBot.prototype.loadPlugin = function (plugin) {
        return __awaiter(this, void 0, void 0, function () {
            var _i, _a, command, error_2;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 3, , 4]);
                        this.logger.info("Loading plugin: ".concat(plugin.name, " v").concat(plugin.version));
                        if (!plugin.initialize) return [3 /*break*/, 2];
                        return [4 /*yield*/, plugin.initialize(this.client, this)];
                    case 1:
                        _b.sent();
                        _b.label = 2;
                    case 2:
                        for (_i = 0, _a = plugin.commands; _i < _a.length; _i++) {
                            command = _a[_i];
                            this.commands.set(command.name, command);
                            this.logger.info("Registered command: ".concat(command.name));
                        }
                        this.plugins.set(plugin.name, plugin);
                        this.logger.info("\u2705 Plugin ".concat(plugin.name, " loaded successfully"));
                        return [3 /*break*/, 4];
                    case 3:
                        error_2 = _b.sent();
                        this.logger.error("Failed to load plugin ".concat(plugin.name, ":"), error_2);
                        throw error_2;
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    ExtensibleBot.prototype.unloadPlugin = function (pluginName) {
        return __awaiter(this, void 0, void 0, function () {
            var plugin, _i, _a, command, error_3;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        plugin = this.plugins.get(pluginName);
                        if (!plugin) {
                            throw new Error("Plugin ".concat(pluginName, " not found"));
                        }
                        _b.label = 1;
                    case 1:
                        _b.trys.push([1, 4, , 5]);
                        if (!plugin.cleanup) return [3 /*break*/, 3];
                        return [4 /*yield*/, plugin.cleanup()];
                    case 2:
                        _b.sent();
                        _b.label = 3;
                    case 3:
                        // Unregister commands
                        for (_i = 0, _a = plugin.commands; _i < _a.length; _i++) {
                            command = _a[_i];
                            this.commands.delete(command.name);
                        }
                        this.plugins.delete(pluginName);
                        this.logger.info("\u2705 Plugin ".concat(pluginName, " unloaded successfully"));
                        return [3 /*break*/, 5];
                    case 4:
                        error_3 = _b.sent();
                        this.logger.error("Failed to unload plugin ".concat(pluginName, ":"), error_3);
                        throw error_3;
                    case 5: return [2 /*return*/];
                }
            });
        });
    };
    ExtensibleBot.prototype.getLoadedPlugins = function () {
        return Array.from(this.plugins.keys());
    };
    ExtensibleBot.prototype.getCommands = function () {
        return Array.from(this.commands.values());
    };
    ExtensibleBot.prototype.start = function () {
        return __awaiter(this, void 0, void 0, function () {
            var error_4;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 2, , 3]);
                        return [4 /*yield*/, this.client.login(this.config.token)];
                    case 1:
                        _a.sent();
                        return [3 /*break*/, 3];
                    case 2:
                        error_4 = _a.sent();
                        this.logger.error("Failed to start bot:", error_4);
                        throw error_4;
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    ExtensibleBot.prototype.shutdown = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _i, _a, _b, name_1, plugin, error_5;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        this.logger.info("Shutting down bot...");
                        _i = 0, _a = this.plugins;
                        _c.label = 1;
                    case 1:
                        if (!(_i < _a.length)) return [3 /*break*/, 6];
                        _b = _a[_i], name_1 = _b[0], plugin = _b[1];
                        if (!plugin.cleanup) return [3 /*break*/, 5];
                        _c.label = 2;
                    case 2:
                        _c.trys.push([2, 4, , 5]);
                        return [4 /*yield*/, plugin.cleanup()];
                    case 3:
                        _c.sent();
                        this.logger.info("Cleaned up plugin: ".concat(name_1));
                        return [3 /*break*/, 5];
                    case 4:
                        error_5 = _c.sent();
                        this.logger.error("Error cleaning up plugin ".concat(name_1, ":"), error_5);
                        return [3 /*break*/, 5];
                    case 5:
                        _i++;
                        return [3 /*break*/, 1];
                    case 6: return [4 /*yield*/, this.client.destroy()];
                    case 7:
                        _c.sent();
                        process.exit(0);
                        return [2 /*return*/];
                }
            });
        });
    };
    return ExtensibleBot;
}());
exports.ExtensibleBot = ExtensibleBot;
