import { Client, Message, EmbedBuilder } from "discord.js";
import type { BotPlugin, BotCommand } from "../types/index.js";
import type { ExtensibleBot } from "../core/Bot.js";
import { Logger } from "../core/util/logger.js";

export class CalculatorPlugin implements BotPlugin {
    name = "CalculatorPlugin";
    description = "Advanced mathematical calculator - responds to =<expression>";
    version = "1.0.0";

    private logger = new Logger();

    commands: BotCommand[] = [
        {
            name: "calc",
            description: "Show calculator help and examples",
            execute: this.showCalculatorHelp.bind(this)
        }
    ];

    // Mathematical constants
    private readonly constants: { [key: string]: number } = {
        'pi': Math.PI,
        'e': Math.E,
        'phi': (1 + Math.sqrt(5)) / 2,  // Golden ratio
        'tau': 2 * Math.PI
    };

    // Mathematical functions
    private readonly functions: { [key: string]: (x: number) => number } = {
        // Trigonometric functions
        'sin': Math.sin,
        'cos': Math.cos,
        'tan': Math.tan,
        'asin': Math.asin,
        'acos': Math.acos,
        'atan': Math.atan,
        'sinh': Math.sinh,
        'cosh': Math.cosh,
        'tanh': Math.tanh,

        // Logarithmic functions
        'ln': Math.log,
        'log': Math.log10,
        'log10': Math.log10,
        'log2': Math.log2,

        // Root functions
        'sqrt': Math.sqrt,
        'cbrt': Math.cbrt,

        // Other functions
        'abs': Math.abs,
        'ceil': Math.ceil,
        'floor': Math.floor,
        'round': Math.round,
        'sign': Math.sign,
        'exp': Math.exp,

        // Convert degrees to radians and vice versa
        'deg': (x) => x * Math.PI / 180,
        'rad': (x) => x * 180 / Math.PI
    };

    // Two-argument functions
    private readonly functions2: { [key: string]: (x: number, y: number) => number } = {
        'pow': Math.pow,
        'atan2': Math.atan2,
        'max': Math.max,
        'min': Math.min,
        'mod': (a, b) => a % b
    };

    async initialize(client: Client, bot: ExtensibleBot): Promise<void> {
        client.on("messageCreate", (message) => {
            this.handleCalculatorRequest(message).catch(error => {
                this.logger.error('Error in calculator processing:', error);
            });
        });

        this.logger.info('CalculatorPlugin initialized - listening for =<expression>');
    }

    async cleanup(): Promise<void> {
        this.logger.info('CalculatorPlugin cleanup completed');
    }

    private async handleCalculatorRequest(message: Message): Promise<void> {
        if (message.author.bot || !message.content.startsWith('=')) return;

        const expression = message.content.substring(1).trim();
        if (!expression) {
            await message.reply('❌ Please provide a mathematical expression after =\n\nExample: `=2+2` or `=sin(pi/4)`');
            return;
        }

        try {
            const result = this.evaluateExpression(expression);
            await this.sendResult(message, expression, result);
        } catch (error) {
            await this.sendError(message, expression, error);
        }
    }

    private evaluateExpression(expression: string): number {
        let cleanExpr = expression
            .toLowerCase()
            .replace(/\s/g, '')  // Remove spaces
            .replace(/×/g, '*')  // Replace × with *
            .replace(/÷/g, '/')  // Replace ÷ with /
            .replace(/\^/g, '**') // Replace ^ with ** for exponentiation
            .replace(/π/g, 'pi') // Replace π with pi
            .replace(/∞/g, 'Infinity'); // Replace ∞ with Infinity

        // Validate expression for safety
        this.validateExpression(cleanExpr);

        // Parse and evaluate
        return this.parseExpression(cleanExpr);
    }

    private validateExpression(expression: string): void {
        // Check for dangerous patterns
        const dangerous = [
            'eval', 'function', 'constructor', 'prototype',
            '__proto__', 'import', 'require', 'process',
            'global', 'this', 'window', 'document'
        ];

        for (const danger of dangerous) {
            if (expression.includes(danger)) {
                throw new Error(`Dangerous operation not allowed: ${danger}`);
            }
        }

        // Check for valid characters only
        const validChars = /^[0-9a-z+\-*/().,^!%\s]*$/;
        if (!validChars.test(expression)) {
            throw new Error('Expression contains invalid characters');
        }

        // Check parentheses balance
        let balance = 0;
        for (const char of expression) {
            if (char === '(') balance++;
            else if (char === ')') balance--;
            if (balance < 0) throw new Error('Mismatched parentheses');
        }
        if (balance !== 0) throw new Error('Unbalanced parentheses');
    }

    private parseExpression(expr: string): number {
        // Replace constants
        for (const [constant, value] of Object.entries(this.constants)) {
            expr = expr.replace(new RegExp('\\b' + constant + '\\b', 'g'), value.toString());
        }

        // Replace single-argument functions
        for (const [funcName, func] of Object.entries(this.functions)) {
            const regex = new RegExp('\\b' + funcName + '\\(([^)]+)\\)', 'g');
            expr = expr.replace(regex, (match, arg) => {
                const argValue = this.parseExpression(arg);
                return func(argValue).toString();
            });
        }

        // Replace two-argument functions
        for (const [funcName, func] of Object.entries(this.functions2)) {
            const regex = new RegExp('\\b' + funcName + '\\(([^,)]+),([^)]+)\\)', 'g');
            expr = expr.replace(regex, (match, arg1, arg2) => {
                const arg1Value = this.parseExpression(arg1);
                const arg2Value = this.parseExpression(arg2);
                return func(arg1Value, arg2Value).toString();
            });
        }

        // Handle factorial
        expr = expr.replace(/(\d+(?:\.\d+)?)!/g, (match, num) => {
            const n = parseFloat(num);
            if (n < 0 || !Number.isInteger(n) || n > 170) {
                throw new Error('Factorial only supports non-negative integers ≤ 170');
            }
            return this.factorial(n).toString();
        });

        // Handle implicit multiplication (like 2pi -> 2*pi)
        expr = expr.replace(/(\d)([a-z])/g, '$1*$2');
        expr = expr.replace(/([a-z])(\d)/g, '$1*$2');
        expr = expr.replace(/(\))(\()/g, '$1*$2');
        expr = expr.replace(/(\d)(\()/g, '$1*$2');
        expr = expr.replace(/(\))(\d)/g, '$1*$2');

        // Evaluate the cleaned expression using Function constructor (safer than eval)
        try {
            const result = new Function('return ' + expr)();

            if (typeof result !== 'number') {
                throw new Error('Expression did not evaluate to a number');
            }

            if (!isFinite(result)) {
                if (isNaN(result)) {
                    throw new Error('Result is not a number (NaN)');
                } else {
                    throw new Error('Result is infinite');
                }
            }

            return result;
        } catch (error) {
            if (error instanceof SyntaxError) {
                throw new Error('Invalid mathematical expression');
            }
            throw error;
        }
    }

    private factorial(n: number): number {
        if (n === 0 || n === 1) return 1;
        let result = 1;
        for (let i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    private async sendResult(message: Message, expression: string, result: number): Promise<void> {
        // Format the result
        let formattedResult: string;

        if (Number.isInteger(result) && Math.abs(result) < 1e15) {
            // Show integers as integers
            formattedResult = result.toString();
        } else if (Math.abs(result) < 1e-10 || Math.abs(result) > 1e15) {
            // Use scientific notation for very small or very large numbers
            formattedResult = result.toExponential(6);
        } else {
            // Use fixed decimal for normal numbers
            formattedResult = parseFloat(result.toFixed(10)).toString();
        }

        const embed = new EmbedBuilder()
            .setTitle("🧮 Calculator Result")
            .setColor(0x00ff00)
            .addFields([
                {
                    name: "Expression",
                    value: `\`${expression}\``,
                    inline: false
                },
                {
                    name: "Result",
                    value: `\`${formattedResult}\``,
                    inline: false
                }
            ])
            .setTimestamp();

        // Add additional info for special results
        if (Math.abs(result - Math.PI) < 1e-10) {
            embed.addFields([{ name: "Note", value: "≈ π (pi)", inline: false }]);
        } else if (Math.abs(result - Math.E) < 1e-10) {
            embed.addFields([{ name: "Note", value: "≈ e (Euler's number)", inline: false }]);
        }

        await message.reply({ embeds: [embed] });
    }

    private async sendError(message: Message, expression: string, error: unknown): Promise<void> {
        const errorMessage = error instanceof Error ? error.message : 'Unknown calculation error';

        const embed = new EmbedBuilder()
            .setTitle("❌ Calculator Error")
            .setColor(0xff0000)
            .addFields([
                {
                    name: "Expression",
                    value: `\`${expression}\``,
                    inline: false
                },
                {
                    name: "Error",
                    value: errorMessage,
                    inline: false
                }
            ])
            .setFooter({ text: "Use !calc for help and examples" })
            .setTimestamp();

        await message.reply({ embeds: [embed] });
    }

    private async showCalculatorHelp(message: Message): Promise<void> {
        const embed = new EmbedBuilder()
            .setTitle("🧮 Advanced Calculator Help")
            .setColor(0x0099ff)
            .setDescription("Type `=<expression>` to calculate mathematical expressions")
            .addFields([
                {
                    name: "Basic Operations",
                    value: "```\n" +
                           "=2+2          → Addition\n" +
                           "=10-3         → Subtraction\n" +
                           "=4*5          → Multiplication\n" +
                           "=20/4         → Division\n" +
                           "=2^3          → Exponentiation\n" +
                           "=17%5         → Modulo\n" +
                           "=5!           → Factorial\n" +
                           "```",
                    inline: false
                },
                {
                    name: "Advanced Functions",
                    value: "```\n" +
                           "=sin(pi/2)    → Sine\n" +
                           "=cos(0)       → Cosine\n" +
                           "=tan(pi/4)    → Tangent\n" +
                           "=sqrt(16)     → Square root\n" +
                           "=cbrt(27)     → Cube root\n" +
                           "=ln(e)        → Natural logarithm\n" +
                           "=log(100)     → Base-10 logarithm\n" +
                           "=abs(-5)      → Absolute value\n" +
                           "```",
                    inline: false
                },
                {
                    name: "Constants",
                    value: "```\n" +
                           "pi   = 3.14159...   π\n" +
                           "e    = 2.71828...   Euler's number\n" +
                           "phi  = 1.61803...   Golden ratio\n" +
                           "tau  = 6.28318...   2π\n" +
                           "```",
                    inline: false
                },
                {
                    name: "Examples",
                    value: "```\n" +
                           "=2*pi*5              → Circle circumference\n" +
                           "=sin(deg(45))        → Sine of 45 degrees\n" +
                           "=pow(2,10)           → 2 to the power of 10\n" +
                           "=(1+sqrt(5))/2       → Golden ratio\n" +
                           "=log2(1024)          → Base-2 logarithm\n" +
                           "```",
                    inline: false
                }
            ])
            .setFooter({
                text: "Supports parentheses, scientific notation, and implicit multiplication (2pi = 2*pi)"
            })
            .setTimestamp();

        await message.reply({ embeds: [embed] });
    }
}