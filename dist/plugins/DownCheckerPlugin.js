import { EmbedBuilder } from "discord.js";
import { Logger } from "../core/util/logger.js";
import fetch from "node-fetch";
export class DownCheckerPlugin {
    name = "DownCheckerPlugin";
    description = "Monitor websites and check if they're down";
    version = "1.0.0";
    logger = new Logger();
    monitoredSites = new Map();
    checkInterval = null;
    client = null;
    commands = [
        {
            name: "monitor",
            description: "Monitor a website for uptime/downtime",
            execute: this.addMonitoring.bind(this)
        },
        {
            name: "unmonitor",
            description: "Stop monitoring a website",
            execute: this.removeMonitoring.bind(this)
        },
        {
            name: "status",
            description: "Check current status of a website or list all monitored sites",
            execute: this.checkStatus.bind(this)
        },
        {
            name: "monitored",
            description: "List all monitored websites",
            execute: this.listMonitored.bind(this)
        }
    ];
    async initialize(client, bot) {
        this.client = client;
        // Start periodic checking every minute
        this.checkInterval = setInterval(() => {
            this.performChecks().catch(error => {
                this.logger.error('Error in website monitoring:', error);
            });
        }, 60000); // Check every minute
        this.logger.info('DownCheckerPlugin initialized - monitoring system started');
    }
    async cleanup() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
        this.logger.info('DownCheckerPlugin cleanup completed');
    }
    async addMonitoring(message, args) {
        if (args.length === 0) {
            const helpEmbed = new EmbedBuilder()
                .setTitle("🔍 Website Monitoring Help")
                .setColor(0x0099ff)
                .setDescription("Monitor websites for uptime and downtime")
                .addFields([
                {
                    name: "Usage",
                    value: "```\n" +
                        "!monitor <url> [interval]    - Monitor a website\n" +
                        "!unmonitor <url>             - Stop monitoring\n" +
                        "!status [url]                - Check status\n" +
                        "!monitored                   - List all monitored sites\n" +
                        "```",
                    inline: false
                },
                {
                    name: "Examples",
                    value: "```\n" +
                        "!monitor https://google.com        - Check every 5 minutes\n" +
                        "!monitor https://example.com 10    - Check every 10 minutes\n" +
                        "!status https://google.com         - Check current status\n" +
                        "```",
                    inline: false
                },
                {
                    name: "Interval Options",
                    value: "• 1-60 minutes (default: 5 minutes)\n• Status updates sent to this channel",
                    inline: false
                }
            ])
                .setFooter({ text: "Monitoring will notify when sites go up/down" });
            await message.reply({ embeds: [helpEmbed] });
            return;
        }
        const url = args[0];
        const intervalMinutes = args[1] ? parseInt(args[1]) : 5;
        // Validate URL
        // @ts-ignore
        if (!this.isValidUrl(url)) {
            await message.reply("❌ Please provide a valid URL (must start with http:// or https://)");
            return;
        }
        // Validate interval
        if (intervalMinutes < 1 || intervalMinutes > 60) {
            await message.reply("❌ Interval must be between 1 and 60 minutes");
            return;
        }
        // @ts-ignore
        const siteKey = this.generateSiteKey(url);
        // Check if already monitoring
        if (this.monitoredSites.has(siteKey)) {
            const site = this.monitoredSites.get(siteKey);
            await message.reply(`⚠️ Already monitoring **${url}** (every ${site.checkInterval} min) in <#${site.channelId}>`);
            return;
        }
        // Perform initial check
        // @ts-ignore
        const initialStatus = await this.checkWebsite(url);
        const monitoredSite = {
            // @ts-ignore
            url,
            channelId: message.channel.id,
            addedBy: message.author.id,
            addedAt: Date.now(),
            checkInterval: intervalMinutes,
            lastStatus: initialStatus.status,
            lastChecked: Date.now(),
            lastUptime: initialStatus.status === 'up' ? Date.now() : 0,
            lastDowntime: initialStatus.status === 'down' ? Date.now() : 0,
            uptimeCount: initialStatus.status === 'up' ? 1 : 0,
            downtimeCount: initialStatus.status === 'down' ? 1 : 0,
            responseTime: initialStatus.responseTime
        };
        this.monitoredSites.set(siteKey, monitoredSite);
        const statusEmoji = initialStatus.status === 'up' ? '✅' : '❌';
        const embed = new EmbedBuilder()
            .setTitle("📊 Website Monitoring Started")
            .setColor(initialStatus.status === 'up' ? 0x00ff00 : 0xff0000)
            .addFields([
            {
                name: "Website",
                value: `[${url}](${url})`,
                inline: true
            },
            {
                name: "Check Interval",
                value: `${intervalMinutes} minutes`,
                inline: true
            },
            {
                name: "Initial Status",
                value: `${statusEmoji} ${initialStatus.status.toUpperCase()}`,
                inline: true
            }
        ]);
        if (initialStatus.responseTime) {
            embed.addFields([{
                    name: "Response Time",
                    value: `${initialStatus.responseTime}ms`,
                    inline: true
                }]);
        }
        embed.addFields([{
                name: "Added By",
                value: `<@${message.author.id}>`,
                inline: true
            }])
            .setTimestamp()
            .setFooter({ text: "Status updates will be posted in this channel" });
        await message.reply({ embeds: [embed] });
    }
    async removeMonitoring(message, args) {
        if (args.length === 0) {
            await message.reply("❌ Please provide the URL to stop monitoring\n\nUsage: `!unmonitor <url>`");
            return;
        }
        const url = args[0];
        // @ts-ignorec
        const siteKey = this.generateSiteKey(url);
        if (!this.monitoredSites.has(siteKey)) {
            await message.reply("❌ Website is not being monitored");
            return;
        }
        const site = this.monitoredSites.get(siteKey);
        // Check permissions (only the person who added it, or admin)
        if (site.addedBy !== message.author.id && !message.member?.permissions.has('Administrator')) {
            await message.reply("❌ Only the person who added this monitor or an administrator can remove it");
            return;
        }
        this.monitoredSites.delete(siteKey);
        const uptime = site.uptimeCount + site.downtimeCount > 0
            ? (site.uptimeCount / (site.uptimeCount + site.downtimeCount) * 100).toFixed(1)
            : '0.0';
        const embed = new EmbedBuilder()
            .setTitle("📊 Monitoring Stopped")
            .setColor(0xff9900)
            .addFields([
            {
                name: "Website",
                value: `[${url}](${url})`,
                inline: false
            },
            {
                name: "Final Statistics",
                value: `**Uptime:** ${uptime}%\n` +
                    `**Total Checks:** ${site.uptimeCount + site.downtimeCount}\n` +
                    `**Up:** ${site.uptimeCount} | **Down:** ${site.downtimeCount}`,
                inline: false
            },
            {
                name: "Monitored For",
                value: this.formatDuration(Date.now() - site.addedAt),
                inline: true
            }
        ])
            .setTimestamp();
        await message.reply({ embeds: [embed] });
    }
    async checkStatus(message, args) {
        if (args.length === 0) {
            // Show all monitored sites
            await this.listMonitored(message, []);
            return;
        }
        const url = args[0];
        // Perform real-time check
        // @ts-ignore
        const result = await this.checkWebsite(url);
        const statusEmoji = result.status === 'up' ? '✅' : '❌';
        const embed = new EmbedBuilder()
            .setTitle("🔍 Website Status Check")
            .setColor(result.status === 'up' ? 0x00ff00 : 0xff0000)
            .addFields([
            {
                name: "Website",
                value: `[${url}](${url})`,
                inline: false
            },
            {
                name: "Status",
                value: `${statusEmoji} ${result.status.toUpperCase()}`,
                inline: true
            }
        ]);
        if (result.responseTime) {
            embed.addFields([{
                    name: "Response Time",
                    value: `${result.responseTime}ms`,
                    inline: true
                }]);
        }
        if (result.error) {
            embed.addFields([{
                    name: "Error Details",
                    value: result.error,
                    inline: false
                }]);
        }
        // Add monitoring info if site is being monitored
        // @ts-ignore
        const siteKey = this.generateSiteKey(url);
        if (this.monitoredSites.has(siteKey)) {
            const site = this.monitoredSites.get(siteKey);
            const uptime = site.uptimeCount + site.downtimeCount > 0
                ? (site.uptimeCount / (site.uptimeCount + site.downtimeCount) * 100).toFixed(1)
                : '0.0';
            embed.addFields([{
                    name: "Monitoring Info",
                    value: `**Uptime:** ${uptime}% (${site.uptimeCount}/${site.uptimeCount + site.downtimeCount})\n` +
                        `**Check Interval:** ${site.checkInterval} minutes\n` +
                        `**Last Checked:** ${this.formatTimestamp(site.lastChecked)}`,
                    inline: false
                }]);
        }
        embed.setTimestamp();
        await message.reply({ embeds: [embed] });
    }
    async listMonitored(message, args) {
        if (this.monitoredSites.size === 0) {
            await message.reply("📝 No websites are currently being monitored.\n\nUse `!monitor <url>` to start monitoring a website.");
            return;
        }
        const embed = new EmbedBuilder()
            .setTitle("📊 Monitored Websites")
            .setColor(0x0099ff)
            .setDescription(`Currently monitoring ${this.monitoredSites.size} website${this.monitoredSites.size === 1 ? '' : 's'}`)
            .setTimestamp();
        let siteIndex = 1;
        for (const [, site] of this.monitoredSites.entries()) {
            const statusEmoji = site.lastStatus === 'up' ? '✅' :
                site.lastStatus === 'down' ? '❌' : '❓';
            const uptime = site.uptimeCount + site.downtimeCount > 0
                ? (site.uptimeCount / (site.uptimeCount + site.downtimeCount) * 100).toFixed(1)
                : '0.0';
            const fieldValue = `${statusEmoji} **Status:** ${site.lastStatus.toUpperCase()}\n` +
                `📊 **Uptime:** ${uptime}%\n` +
                `⏱️ **Interval:** ${site.checkInterval}m\n` +
                `👤 **Added by:** <@${site.addedBy}>\n` +
                `📝 **Channel:** <#${site.channelId}>`;
            embed.addFields([{
                    name: `${siteIndex}. ${site.url}`,
                    value: fieldValue,
                    inline: true
                }]);
            siteIndex++;
            // Discord has a limit on embed fields
            if (siteIndex > 25) {
                embed.setFooter({ text: `Showing first 25 of ${this.monitoredSites.size} monitored sites` });
                break;
            }
        }
        await message.reply({ embeds: [embed] });
    }
    async performChecks() {
        const now = Date.now();
        for (const [siteKey, site] of this.monitoredSites.entries()) {
            // Check if it's time to check this site
            const timeSinceLastCheck = now - site.lastChecked;
            const checkIntervalMs = site.checkInterval * 60 * 1000;
            if (timeSinceLastCheck >= checkIntervalMs) {
                try {
                    const result = await this.checkWebsite(site.url);
                    const previousStatus = site.lastStatus;
                    // Update site info
                    site.lastChecked = now;
                    site.lastStatus = result.status;
                    site.responseTime = result.responseTime;
                    if (result.status === 'up') {
                        site.uptimeCount++;
                        site.lastUptime = now;
                    }
                    else {
                        site.downtimeCount++;
                        site.lastDowntime = now;
                    }
                    // Send notification if status changed
                    if (previousStatus !== result.status && previousStatus !== 'unknown') {
                        await this.sendStatusNotification(site, result, previousStatus);
                    }
                }
                catch (error) {
                    this.logger.error(`Error checking ${site.url}:`, error);
                }
            }
        }
    }
    async checkWebsite(url) {
        const startTime = Date.now();
        try {
            const response = await fetch(url, {
                method: 'HEAD',
                headers: {
                    'User-Agent': 'Mozilla/5.0 (compatible; Discord-Bot-Monitor/1.0)'
                }
            });
            const responseTime = Date.now() - startTime;
            if (response.status >= 200 && response.status < 400) {
                return { status: 'up', responseTime };
            }
            else {
                return {
                    status: 'down',
                    responseTime,
                    error: `HTTP ${response.status} ${response.statusText}`
                };
            }
        }
        catch (error) {
            const responseTime = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            return {
                status: 'down',
                responseTime: responseTime < 1000 ? responseTime : undefined,
                error: errorMessage
            };
        }
    }
    async sendStatusNotification(site, result, previousStatus) {
        if (!this.client)
            return;
        try {
            const channel = await this.client.channels.fetch(site.channelId);
            if (!channel)
                return;
            const isUp = result.status === 'up';
            const statusEmoji = isUp ? '✅' : '❌';
            const color = isUp ? 0x00ff00 : 0xff0000;
            const uptime = site.uptimeCount + site.downtimeCount > 0
                ? (site.uptimeCount / (site.uptimeCount + site.downtimeCount) * 100).toFixed(1)
                : '0.0';
            const embed = new EmbedBuilder()
                .setTitle(`${statusEmoji} Website Status Change`)
                .setColor(color)
                .addFields([
                {
                    name: "Website",
                    value: `[${site.url}](${site.url})`,
                    inline: false
                },
                {
                    name: "Status Change",
                    value: `${previousStatus.toUpperCase()} → **${result.status.toUpperCase()}**`,
                    inline: true
                },
                {
                    name: "Current Uptime",
                    value: `${uptime}%`,
                    inline: true
                }
            ]);
            if (result.responseTime) {
                embed.addFields([{
                        name: "Response Time",
                        value: `${result.responseTime}ms`,
                        inline: true
                    }]);
            }
            if (result.error) {
                embed.addFields([{
                        name: "Error Details",
                        value: result.error,
                        inline: false
                    }]);
            }
            embed.setTimestamp()
                .setFooter({ text: `Monitoring interval: ${site.checkInterval} minutes` });
            await channel.send({ embeds: [embed] });
        }
        catch (error) {
            this.logger.error(`Failed to send status notification for ${site.url}:`, error);
        }
    }
    isValidUrl(url) {
        try {
            const urlObj = new URL(url);
            return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
        }
        catch {
            return false;
        }
    }
    generateSiteKey(url) {
        // Normalize URL for consistent key generation
        try {
            const urlObj = new URL(url);
            return `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`.toLowerCase();
        }
        catch {
            return url.toLowerCase();
        }
    }
    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        if (days > 0)
            return `${days}d ${hours % 24}h`;
        if (hours > 0)
            return `${hours}h ${minutes % 60}m`;
        if (minutes > 0)
            return `${minutes}m ${seconds % 60}s`;
        return `${seconds}s`;
    }
    formatTimestamp(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        if (minutes < 1)
            return 'Just now';
        if (minutes < 60)
            return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24)
            return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }
}
//# sourceMappingURL=DownCheckerPlugin.js.map