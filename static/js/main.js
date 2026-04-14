const UI_TRANSLATION_CACHE_KEY = "uiTranslationCacheV1";
let deferredInstallPrompt = null;

function getCurrentLanguage() {
  return localStorage.getItem("userLanguage") || "en";
}

function getTranslationCache() {
  try {
    return JSON.parse(localStorage.getItem(UI_TRANSLATION_CACHE_KEY) || "{}");
  } catch {
    return {};
  }
}

function setTranslationCache(cache) {
  localStorage.setItem(UI_TRANSLATION_CACHE_KEY, JSON.stringify(cache));
}

function compressAndResizeImage(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error("No file provided"));
      return;
    }

    const img = new Image();
    const reader = new FileReader();

    reader.onload = () => {
      img.src = reader.result;
    };

    reader.onerror = () => {
      reject(new Error("Failed to read image file"));
    };

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas context unavailable"));
        return;
      }

      // Model expects a fixed 224x224 image.
      ctx.drawImage(img, 0, 0, 224, 224);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Failed to create compressed image blob"));
            return;
          }
          resolve(blob);
        },
        "image/jpeg",
        0.7
      );
    };

    img.onerror = () => {
      reject(new Error("Invalid image content"));
    };

    reader.readAsDataURL(file);
  });
}

function collectI18nElements() {
  const textEls = Array.from(document.querySelectorAll("[data-i18n]"));
  const placeholderEls = Array.from(
    document.querySelectorAll("[data-i18n-placeholder]")
  );

  textEls.forEach((el) => {
    if (!el.dataset.en) {
      el.dataset.en = el.textContent.trim();
    }
  });

  placeholderEls.forEach((el) => {
    if (!el.dataset.enPlaceholder) {
      el.dataset.enPlaceholder = el.getAttribute("placeholder") || "";
    }
  });

  return { textEls, placeholderEls };
}

async function fetchUiTranslations(missingTexts, language) {
  if (!missingTexts.length || language === "en") {
    return {};
  }

  const response = await fetch("/translate-ui", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language, texts: missingTexts }),
  });

  if (!response.ok) {
    throw new Error("translate-ui failed");
  }

  const payload = await response.json();
  return payload.translations || {};
}

async function applyLanguage(language) {
  const { textEls, placeholderEls } = collectI18nElements();
  const hiddenLangInput = document.getElementById("selectedLanguage");
  const langSelect = document.getElementById("language-select");
  const mobileLangSelect = document.getElementById("language-select-mobile");

  localStorage.setItem("userLanguage", language);
  if (langSelect) langSelect.value = language;
  if (mobileLangSelect) mobileLangSelect.value = language;
  if (hiddenLangInput) hiddenLangInput.value = language;

  if (language === "en") {
    textEls.forEach((el) => {
      el.textContent = el.dataset.en || el.textContent;
    });
    placeholderEls.forEach((el) => {
      el.setAttribute("placeholder", el.dataset.enPlaceholder || "");
    });
    window.dispatchEvent(new CustomEvent("app:language-changed", { detail: { language } }));
    return;
  }

  const cache = getTranslationCache();
  const langCache = cache[language] || {};
  const allEnglish = [
    ...textEls.map((el) => el.dataset.en || ""),
    ...placeholderEls.map((el) => el.dataset.enPlaceholder || ""),
  ].filter(Boolean);
  const uniqueEnglish = [...new Set(allEnglish)];
  const missingTexts = uniqueEnglish.filter((txt) => !langCache[txt]);

  if (missingTexts.length) {
    try {
      const translated = await fetchUiTranslations(missingTexts, language);
      cache[language] = { ...langCache, ...translated };
      setTranslationCache(cache);
    } catch (error) {
      console.error("UI translation request failed:", error);
    }
  }

  const finalCache = (getTranslationCache()[language] || {});

  textEls.forEach((el) => {
    const en = el.dataset.en || "";
    el.textContent = finalCache[en] || en;
  });

  placeholderEls.forEach((el) => {
    const en = el.dataset.enPlaceholder || "";
    el.setAttribute("placeholder", finalCache[en] || en);
  });

  window.dispatchEvent(new CustomEvent("app:language-changed", { detail: { language } }));
}

window.applyLanguage = applyLanguage;
window.getCurrentLanguage = getCurrentLanguage;
window.compressAndResizeImage = compressAndResizeImage;

function wireLanguageSelector() {
  const langSelect = document.getElementById("language-select");
  const mobileLangSelect = document.getElementById("language-select-mobile");

  if (langSelect) {
    langSelect.addEventListener("change", (e) => applyLanguage(e.target.value));
  }

  if (mobileLangSelect) {
    mobileLangSelect.addEventListener("change", (e) => applyLanguage(e.target.value));
  }
}

function wireMobileMenu() {
  const navLinks = document.getElementById("navLinks");

  window.showMenu = function () {
    if (navLinks) navLinks.style.right = "0";
  };

  window.hideMenu = function () {
    if (navLinks) navLinks.style.right = "-240px";
  };
}

function wireUploadModal() {
  const uploadModal = document.getElementById("uploadModal");
  const uploadBtn = document.getElementById("uploadBtn");
  const cropInput = document.getElementById("selectedCrop");
  const imageInput = document.getElementById("imageInput");
  const uploadForm = document.querySelector("form[action='/upload']");

  if (!uploadModal || !uploadBtn || !cropInput || !imageInput) {
    return;
  }

  const statusText = document.createElement("div");
  statusText.style.marginTop = "10px";
  statusText.style.fontSize = "14px";
  statusText.style.color = "#2e7d32";
  imageInput.parentNode?.appendChild(statusText);

  window.openUploadModal = async function () {
    if (
      window.requireAuthForAction &&
      !(await window.requireAuthForAction({
        reason: "Please sign in before uploading crop images.",
      }))
    ) {
      return;
    }
    uploadModal.style.display = "flex";
  };

  window.closeUploadModal = function () {
    uploadModal.style.display = "none";
    cropInput.value = "";
    imageInput.value = "";
    uploadBtn.disabled = true;
    statusText.textContent = "";
    document.querySelectorAll(".modal-item").forEach((item) => item.classList.remove("selected"));
  };

  window.selectCrop = function (element) {
    document.querySelectorAll(".modal-item").forEach((item) => item.classList.remove("selected"));
    element.classList.add("selected");
    cropInput.value =
      element.getAttribute("data-crop") || element.querySelector("p")?.innerText.trim() || "";
    imageInput.click();
  };

  imageInput.addEventListener("change", () => {
    if (imageInput.files.length > 0) {
      uploadBtn.disabled = false;
      statusText.textContent = `Selected file: ${imageInput.files[0].name}`;
    } else {
      uploadBtn.disabled = true;
      statusText.textContent = "";
    }
  });

  uploadForm?.addEventListener("submit", () => {
    uploadBtn.disabled = true;
    statusText.textContent = "Uploading image... AI is analyzing the leaf.";
  });
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((error) => {
      console.error("Service worker registration failed:", error);
    });
  });
}

function isRunningStandalone() {
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    window.navigator.standalone === true
  );
}

function isIOSDevice() {
  return /iphone|ipad|ipod/i.test(window.navigator.userAgent);
}

function shouldShowInstallPopup() {
  return !isRunningStandalone();
}

function closeInstallPopup() {
  const popup = document.getElementById("installPromptPopup");
  if (!popup) return;
  popup.classList.remove("show");
  setTimeout(() => popup.remove(), 180);
}

function createInstallPopup() {
  if (!shouldShowInstallPopup() || document.getElementById("installPromptPopup")) {
    return;
  }

  const isIOS = isIOSDevice();
  const supportsNativePrompt = !!deferredInstallPrompt;

  const popup = document.createElement("div");
  popup.id = "installPromptPopup";
  popup.className = "install-popup";

  const bodyHtml = supportsNativePrompt
    ? `
      <p class="install-popup-text" data-i18n="install_popup_text">
        Install CropGuard AI on your phone for one-tap access in the field.
      </p>
      <div class="install-popup-actions">
        <button type="button" class="install-dismiss-btn" id="installLaterBtn" data-i18n="install_later">Later</button>
        <button type="button" class="install-now-btn" id="installNowBtn" data-i18n="install_now">Install App</button>
      </div>
    `
    : isIOS
    ? `
      <p class="install-popup-text" data-i18n="install_ios_text">
        Add this app to your home screen: tap Share, then tap Add to Home Screen.
      </p>
      <div class="install-popup-actions single">
        <button type="button" class="install-dismiss-btn" id="installLaterBtn" data-i18n="ok_btn">OK</button>
      </div>
    `
    : `
      <p class="install-popup-text" data-i18n="install_popup_unavailable">
        To install this app, open browser menu and tap Add to Home Screen.
      </p>
      <div class="install-popup-actions single">
        <button type="button" class="install-dismiss-btn" id="installLaterBtn" data-i18n="ok_btn">OK</button>
      </div>
    `;

  popup.innerHTML = `
    <div class="install-popup-card">
      <div class="install-popup-header">
        <i class="fa-solid fa-mobile-screen-button"></i>
        <h4 data-i18n="install_popup_title">Install CropGuard AI</h4>
        <button type="button" class="install-close-btn" id="installCloseBtn" aria-label="Close">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>
      ${bodyHtml}
    </div>
  `;

  document.body.appendChild(popup);
  requestAnimationFrame(() => popup.classList.add("show"));
  if (typeof applyLanguage === "function") {
    applyLanguage(getCurrentLanguage()).catch(() => {});
  }

  const closeBtn = document.getElementById("installCloseBtn");
  const laterBtn = document.getElementById("installLaterBtn");
  const nowBtn = document.getElementById("installNowBtn");

  const dismiss = () => {
    closeInstallPopup();
  };

  closeBtn?.addEventListener("click", dismiss);
  laterBtn?.addEventListener("click", dismiss);

  if (nowBtn) {
    nowBtn.addEventListener("click", async () => {
      if (!deferredInstallPrompt) return;
      deferredInstallPrompt.prompt();
      try {
        await deferredInstallPrompt.userChoice;
      } catch (error) {
        console.error("Install prompt failed:", error);
      }
      deferredInstallPrompt = null;
      closeInstallPopup();
    });
  }

  popup.addEventListener("click", (event) => {
    if (event.target === popup) dismiss();
  });
}

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredInstallPrompt = event;
  createInstallPopup();
});

window.addEventListener("appinstalled", () => {
  deferredInstallPrompt = null;
  closeInstallPopup();
});

document.addEventListener("DOMContentLoaded", async () => {
  wireLanguageSelector();
  wireMobileMenu();
  wireUploadModal();
  registerServiceWorker();
  setTimeout(createInstallPopup, 1200);

  // Restore previously selected language on each page load.
  const savedLang = localStorage.getItem("userLanguage") || "en";
  const langSelect = document.getElementById("language-select");
  const mobileLangSelect = document.getElementById("language-select-mobile");
  if (langSelect) langSelect.value = savedLang;
  if (mobileLangSelect) mobileLangSelect.value = savedLang;
  await applyLanguage(savedLang);
});
