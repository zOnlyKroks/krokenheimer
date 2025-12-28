// Utility functions and helpers
use anyhow::Result;
use std::path::Path;

pub struct FileUtils;

impl FileUtils {
    /// Ensure directory exists, create if not
    pub fn ensure_directory(path: &str) -> Result<()> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    /// Get file size in bytes
    pub fn get_file_size(path: &str) -> Result<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }

    /// Check if path exists and is a directory
    pub fn is_directory(path: &str) -> bool {
        Path::new(path).is_dir()
    }

    /// Check if path exists and is a file
    pub fn is_file(path: &str) -> bool {
        Path::new(path).is_file()
    }

    /// Get directory contents
    pub fn list_directory(path: &str) -> Result<Vec<String>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }
        Ok(entries)
    }
}

pub struct TextUtils;

impl TextUtils {
    /// Clean text for training (remove excessive whitespace, etc.)
    pub fn clean_text(text: &str) -> String {
        // Remove excessive whitespace
        let cleaned = text.trim()
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // Normalize unicode
        cleaned.chars()
            .map(|c| match c {
                '\u{2019}' => '\'', // Smart quote to regular quote
                '\u{201C}' | '\u{201D}' => '"', // Smart quotes to regular quotes
                '\u{2013}' | '\u{2014}' => '-', // En/em dash to hyphen
                c => c,
            })
            .collect()
    }

    /// Truncate text to approximate character limit
    pub fn truncate_text(text: &str, max_chars: usize) -> String {
        if text.len() <= max_chars {
            return text.to_string();
        }

        // Try to truncate at word boundary
        if let Some(truncate_pos) = text[..max_chars].rfind(' ') {
            format!("{}...", &text[..truncate_pos])
        } else {
            format!("{}...", &text[..max_chars.saturating_sub(3)])
        }
    }

    /// Count approximate words in text
    pub fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Estimate reading time in seconds
    pub fn estimate_reading_time(text: &str) -> u32 {
        let words = Self::count_words(text);
        // Assume 200 words per minute reading speed
        ((words as f32 / 200.0) * 60.0) as u32
    }
}

pub struct PerformanceUtils;

impl PerformanceUtils {
    /// Get current timestamp in milliseconds
    pub fn now_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Format duration in human readable format
    pub fn format_duration(start_millis: u64, end_millis: u64) -> String {
        let duration_ms = end_millis.saturating_sub(start_millis);
        let seconds = duration_ms as f64 / 1000.0;

        if seconds < 1.0 {
            format!("{:.0}ms", duration_ms)
        } else if seconds < 60.0 {
            format!("{:.1}s", seconds)
        } else {
            let minutes = seconds / 60.0;
            format!("{:.1}m", minutes)
        }
    }

    /// Get memory usage estimate (simplified)
    pub fn estimate_memory_usage() -> Result<u64> {
        // This is a simplified implementation
        // In a real implementation, you'd use system APIs
        Ok(0)
    }
}

pub struct ConfigUtils;

impl ConfigUtils {
    /// Load JSON configuration from file
    pub fn load_json_config<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
        let content = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to JSON file
    pub fn save_json_config<T: serde::Serialize>(config: &T, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get environment variable with default value
    pub fn get_env_var(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }
}