// UI translation helper with localStorage caching.

class UITranslator {
    constructor() {
        this.currentLanguage = localStorage.getItem("selected_language") || "en";
        this.translationCache = this.loadCache();
        this.pendingTranslations = new Set();
    }

    // Read cached translations safely.
    loadCache() {
        try {
            const cache = localStorage.getItem("translationCache");
            return cache ? JSON.parse(cache) : {};
        } catch (e) {
            window.CropGuardLogger?.error("Failed to load translation cache:", e);
            return {};
        }
    }

    // Persist cache for future page loads.
    saveCache() {
        try {
            localStorage.setItem("translationCache", JSON.stringify(this.translationCache));
        } catch (e) {
            window.CropGuardLogger?.error("Failed to save translation cache:", e);
        }
    }

    // Unique cache key per page-language pair.
    getCacheKey(page, language) {
        return `${page}_${language}`;
    }

    // Request fresh translations from the backend.
    async fetchTranslations(page, language) {
        // Avoid duplicate requests for the same page/language.
        const requestKey = this.getCacheKey(page, language);
        if (this.pendingTranslations.has(requestKey)) {
            return null;
        }

        this.pendingTranslations.add(requestKey);

        try {
            const response = await fetch("/translate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ page, language }),
            });

            if (!response.ok) {
                throw new Error(`Translation request failed: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Cache result with timestamp for expiry checks.
            const cacheKey = this.getCacheKey(page, language);
            this.translationCache[cacheKey] = {
                translations: data.translations,
                timestamp: Date.now(),
            };
            
            this.saveCache();
            this.pendingTranslations.delete(requestKey);
            
            return data.translations;
        } catch (error) {
            window.CropGuardLogger?.error("Translation fetch error:", error);
            this.pendingTranslations.delete(requestKey);
            return null;
        }
    }

    // Return cached translations if valid, otherwise fetch.
    async getTranslations(page, language) {
        // English is the default source content.
        if (language === "en") {
            return {};
        }

        const cacheKey = this.getCacheKey(page, language);
        const cached = this.translationCache[cacheKey];

        // Keep cached data for up to 7 days.
        const CACHE_EXPIRY = 7 * 24 * 60 * 60 * 1000; // 7 days
        if (cached && (Date.now() - cached.timestamp < CACHE_EXPIRY)) {
            return cached.translations;
        }

        return await this.fetchTranslations(page, language);
    }

    // Apply translated labels and placeholders to the page.
    applyTranslations(translations) {
        if (!translations || Object.keys(translations).length === 0) {
            return;
        }

        const elements = document.querySelectorAll("[data-translate]");
        
        elements.forEach(element => {
            const key = element.getAttribute("data-translate");
            const translatedText = translations[key];
            
            if (translatedText) {
                // Keep original text so we can restore English instantly.
                if (!element.hasAttribute("data-original")) {
                    element.setAttribute("data-original", element.textContent.trim());
                }
                
                element.textContent = translatedText;
            }
        });

        const placeholderElements = document.querySelectorAll("[data-translate-placeholder]");
        
        placeholderElements.forEach(element => {
            const key = element.getAttribute("data-translate-placeholder");
            const translatedText = translations[key];
            
            if (translatedText) {
                if (!element.hasAttribute("data-original-placeholder")) {
                    element.setAttribute("data-original-placeholder", element.placeholder);
                }
                
                element.placeholder = translatedText;
            }
        });
    }

    // Restore original English content from stored attributes.
    restoreOriginalText() {
        const elements = document.querySelectorAll("[data-original]");
        
        elements.forEach(element => {
            const originalText = element.getAttribute("data-original");
            
            if (originalText) {
                element.textContent = originalText;
            }
        });

        const placeholderElements = document.querySelectorAll("[data-original-placeholder]");
        
        placeholderElements.forEach(element => {
            const originalPlaceholder = element.getAttribute("data-original-placeholder");
            
            if (originalPlaceholder) {
                element.placeholder = originalPlaceholder;
            }
        });
    }

    // Translate current page to the selected language.
    async translatePage(page, language) {
        this.currentLanguage = language;
        
        if (language === "en") {
            this.restoreOriginalText();
            return;
        }

        const loadingIndicator = document.getElementById("translation-loading");
        if (loadingIndicator) {
            loadingIndicator.style.display = "block";
        }

        try {
            const translations = await this.getTranslations(page, language);
            
            if (translations) {
                this.applyTranslations(translations);
            }
        } finally {
            if (loadingIndicator) {
                loadingIndicator.style.display = "none";
            }
        }
    }

    // Clear local translation cache.
    clearCache() {
        this.translationCache = {};
        localStorage.removeItem("translationCache");
    }
}

// Shared translator instance.
const translator = new UITranslator();

document.addEventListener("DOMContentLoaded", () => {
    const savedLanguage = localStorage.getItem("selected_language") || "en";
    
    const currentPage = determinePage();
    
    if (savedLanguage !== "en") {
        translator.translatePage(currentPage, savedLanguage);
    }
});

// Infer page key from URL path.
function determinePage() {
    const path = window.location.pathname;
    
    if (path.includes("/about")) return "about";
    if (path.includes("/guide")) return "guide";
    if (path.includes("/result") || path === "/upload") return "result";
    
    return "home";
}

// Global language switcher used by header controls.
window.setLanguageWithTranslation = async function(lang) {
    localStorage.setItem("selected_language", lang);
    
    const langSelect = document.getElementById("languageSelect");
    const mobileLangSelect = document.getElementById("mobileLangSelect");
    const langInput = document.getElementById("selectedLanguage");
    
    if (langSelect) langSelect.value = lang;
    if (mobileLangSelect) mobileLangSelect.value = lang;
    if (langInput) langInput.value = lang;
    
    const navLinks = document.getElementById("navLinks");
    if (navLinks) {
        navLinks.style.right = "-200px";
    }
    
    const currentPage = determinePage();
    await translator.translatePage(currentPage, lang);
};

// Expose for manual debugging in browser console.
window.translator = translator;
