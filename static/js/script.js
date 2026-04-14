// Initialize page behaviors once DOM is ready.
document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page || "home";

  // Restore saved language across pages.
  const savedLanguage = localStorage.getItem("userLanguage") || "en";
  const langSelect = document.getElementById("languageSelect");
  const mobileLangSelect = document.getElementById("mobileLangSelect");

  if (langSelect) {
    langSelect.value = savedLanguage;
  }
  if (mobileLangSelect) {
    mobileLangSelect.value = savedLanguage;
  }

  // Keep hidden form input in sync with selected language.
  const langInput = document.getElementById("selectedLanguage");
  if (langInput) {
    langInput.value = savedLanguage;
  }

  // Update language in selectors and trigger translation.
  window.setLanguage = function (lang) {
    localStorage.setItem("userLanguage", lang);

    const topSelect = document.getElementById("languageSelect");
    const mobileSelect = document.getElementById("mobileLangSelect");
    const hiddenInput = document.getElementById("selectedLanguage");

    if (topSelect) topSelect.value = lang;
    if (mobileSelect) mobileSelect.value = lang;
    if (hiddenInput) hiddenInput.value = lang;

    applyPageTranslations(page, lang);

    // Close mobile drawer after language change.
    const navLinksMenu = document.getElementById("navLinks");
    if (navLinksMenu) {
      navLinksMenu.style.right = "-200px";
    }
  };

  // Mobile nav open/close handlers.
  const navLinks = document.getElementById("navLinks");

  window.showMenu = function () {
    if (navLinks) navLinks.style.right = "0";
  };

  window.hideMenu = function () {
    if (navLinks) navLinks.style.right = "-200px";
  };

  // Upload modal interactions.
  let selectedCrop = null;

  const uploadModal = document.getElementById("uploadModal");
  const uploadBtn = document.getElementById("uploadBtn");
  const cropInput = document.getElementById("selectedCrop");
  const imageInput = document.getElementById("imageInput");
  const uploadForm = document.querySelector("form[action='/upload']");

  const statusText = document.createElement("div");
  statusText.style.marginTop = "10px";
  statusText.style.fontSize = "14px";
  statusText.style.color = "#2e7d32";

  if (imageInput && imageInput.parentNode) {
    imageInput.parentNode.appendChild(statusText);
  }

  window.openUploadModal = function () {
    if (uploadModal) uploadModal.style.display = "flex";
  };

  window.closeUploadModal = function () {
    if (uploadModal) uploadModal.style.display = "none";
    resetSelection();
  };

  window.selectCrop = function (element) {
    document
      .querySelectorAll(".modal-item")
      .forEach((item) => item.classList.remove("selected"));

    element.classList.add("selected");
    selectedCrop = element.querySelector("p")?.innerText || "";

    if (cropInput) cropInput.value = selectedCrop;
    if (uploadBtn) uploadBtn.disabled = true;

    if (imageInput) imageInput.click();
  };

  function resetSelection() {
    selectedCrop = null;
    if (uploadBtn) uploadBtn.disabled = true;
    if (cropInput) cropInput.value = "";
    if (imageInput) imageInput.value = "";
    statusText.innerText = "";
  }

  if (imageInput) {
    imageInput.addEventListener("change", () => {
      if (imageInput.files.length > 0) {
        statusText.innerText = "Selected file: " + imageInput.files[0].name;
        if (uploadBtn) uploadBtn.disabled = false;
      } else {
        statusText.innerText = "";
        if (uploadBtn) uploadBtn.disabled = true;
      }
    });
  }

  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      if (uploadBtn) uploadBtn.disabled = true;
      statusText.innerText = "Uploading image... AI is analyzing the leaf";
    });
  }

  applyPageTranslations(page, savedLanguage);
});

async function applyPageTranslations(page, lang) {
  try {
    if (lang === "en") {
      restoreDefaultEnglish();
      return;
    }

    const response = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page, language: lang }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch translations");
    }

    const data = await response.json();
    const translations = data.translations || {};

    document.querySelectorAll("[data-translate]").forEach((el) => {
      const key = el.getAttribute("data-translate");
      if (translations[key]) {
        el.textContent = translations[key];
      }
    });

    document.querySelectorAll("[data-translate-placeholder]").forEach((el) => {
      const key = el.getAttribute("data-translate-placeholder");
      if (translations[key]) {
        el.placeholder = translations[key];
      }
    });
  } catch (error) {
    console.error("Translation error:", error);
  }
}

function restoreDefaultEnglish() {
  document.querySelectorAll("[data-translate]").forEach((el) => {
    const original = el.getAttribute("data-default");
    if (original) {
      el.textContent = original;
    }
  });

  document.querySelectorAll("[data-translate-placeholder]").forEach((el) => {
    const original = el.getAttribute("data-default-placeholder");
    if (original) {
      el.placeholder = original;
    }
  });
}
