const AUTH_PROMPT_SEEN_KEY = "cg_auth_prompt_seen";

let firebaseApp = null;
let firebaseAuth = null;
let authMode = "signin";
let authModalBlocking = false;

function initFirebaseAuth() {
  if (!window.firebase || !window.CROPGUARD_FIREBASE_CONFIG) {
    return false;
  }

  if (!firebaseApp) {
    firebaseApp = window.firebase.apps.length
      ? window.firebase.app()
      : window.firebase.initializeApp(window.CROPGUARD_FIREBASE_CONFIG);
  }

  if (!firebaseAuth) {
    firebaseAuth = window.firebase.auth(firebaseApp);
  }

  return true;
}

function getAuthInstance() {
  return initFirebaseAuth() ? firebaseAuth : null;
}

function getAuthModalElements() {
  return {
    modal: document.getElementById("authModal"),
    closeBtn: document.getElementById("authModalClose"),
    title: document.getElementById("authModalTitle"),
    text: document.getElementById("authModalText"),
    error: document.getElementById("authError"),
    form: document.getElementById("authForm"),
    email: document.getElementById("authEmail"),
    password: document.getElementById("authPassword"),
    confirmWrap: document.getElementById("confirmPasswordWrap"),
    confirmPassword: document.getElementById("authConfirmPassword"),
    submit: document.getElementById("authSubmitBtn"),
    googleBtn: document.getElementById("googleAuthBtn"),
    signinTab: document.getElementById("signinTab"),
    signupTab: document.getElementById("signupTab"),
    backdrop: document.getElementById("authModalBackdrop"),
  };
}

function setAuthError(message = "") {
  const { error } = getAuthModalElements();
  if (!error) return;

  if (message) {
    error.hidden = false;
    error.textContent = message;
    return;
  }

  error.hidden = true;
  error.textContent = "";
}

function setAuthMode(mode) {
  authMode = mode === "signup" ? "signup" : "signin";
  const {
    signinTab,
    signupTab,
    title,
    text,
    submit,
    password,
    confirmWrap,
    confirmPassword,
  } = getAuthModalElements();

  signinTab?.classList.toggle("active", authMode === "signin");
  signupTab?.classList.toggle("active", authMode === "signup");

  if (title) {
    title.dataset.en =
      authMode === "signup" ? "Create your CropGuard AI account" : "Sign in to CropGuard AI";
    title.textContent = title.dataset.en;
  }

  if (text) {
    text.dataset.en =
      authMode === "signup"
        ? "Create your account to upload crop images, save access, and continue diagnosis securely."
        : "Sign in to upload images, chat with the assistant, and continue your crop diagnosis.";
    text.textContent = text.dataset.en;
  }

  if (submit) {
    submit.dataset.en = authMode === "signup" ? "Create Account" : "Continue";
    submit.textContent = submit.dataset.en;
  }

  if (password) {
    password.autocomplete = authMode === "signup" ? "new-password" : "current-password";
  }

  if (confirmWrap) {
    confirmWrap.hidden = authMode !== "signup";
  }

  if (confirmPassword) {
    confirmPassword.required = authMode === "signup";
    if (authMode !== "signup") {
      confirmPassword.value = "";
    }
  }

  setAuthError("");
  if (typeof window.applyLanguage === "function") {
    window.applyLanguage(window.getCurrentLanguage?.() || "en").catch(() => {});
  }
}

function openAuthModal({ blocking = false, reason = "" } = {}) {
  const { modal, closeBtn, text } = getAuthModalElements();
  if (!modal) return;

  authModalBlocking = blocking;
  modal.hidden = false;
  modal.classList.add("show");
  modal.dataset.blocking = blocking ? "true" : "false";

  if (closeBtn) {
    closeBtn.hidden = blocking;
  }

  if (reason && text) {
    text.textContent = reason;
  } else {
    setAuthMode(authMode);
  }

  document.body.classList.toggle("auth-modal-open", true);
}

function closeAuthModal(force = false) {
  const { modal } = getAuthModalElements();
  if (!modal) return;
  if (authModalBlocking && !force) return;

  modal.classList.remove("show");
  modal.hidden = true;
  document.body.classList.toggle("auth-modal-open", false);
  setAuthError("");
}

function updateAuthButtons(user) {
  const desktopUser = document.getElementById("auth-user-desktop");
  const mobileUser = document.getElementById("auth-user-mobile");
  const desktopOpen = document.getElementById("auth-open-desktop");
  const mobileOpen = document.getElementById("auth-open-mobile");
  const desktopName = document.getElementById("auth-user-desktop-name");
  const mobileName = document.getElementById("auth-user-mobile-name");

  if (!user) {
    desktopUser?.setAttribute("hidden", "hidden");
    mobileUser?.setAttribute("hidden", "hidden");
    desktopOpen?.removeAttribute("hidden");
    mobileOpen?.removeAttribute("hidden");
    return;
  }

  const displayName =
    user.displayName || user.email?.split("@")[0] || "Farmer";

  if (desktopName) desktopName.textContent = displayName;
  if (mobileName) mobileName.textContent = displayName;

  desktopOpen?.setAttribute("hidden", "hidden");
  mobileOpen?.setAttribute("hidden", "hidden");
  desktopUser?.removeAttribute("hidden");
  mobileUser?.removeAttribute("hidden");
}

async function syncAuthTokens() {
  const auth = getAuthInstance();
  const user = auth?.currentUser || null;
  const token = user ? await user.getIdToken() : "";

  document.querySelectorAll('input[name="auth_token"]').forEach((input) => {
    input.value = token;
  });

  return token;
}

async function submitProtectedForm(form) {
  if (!form) return false;

  const allowed = await window.requireAuthForAction({
    reason: "Please sign in to continue with crop analysis.",
  });
  if (!allowed) return false;

  await syncAuthTokens();
  HTMLFormElement.prototype.submit.call(form);
  return true;
}

async function handleEmailAuth(event) {
  event.preventDefault();
  const auth = getAuthInstance();
  if (!auth) return;

  const { email, password, confirmPassword } = getAuthModalElements();
  const emailValue = email?.value.trim() || "";
  const passwordValue = password?.value || "";
  const confirmValue = confirmPassword?.value || "";

  if (!emailValue || !passwordValue) {
    setAuthError("Please enter your email and password.");
    return;
  }

  if (authMode === "signup" && passwordValue !== confirmValue) {
    setAuthError("Passwords do not match.");
    return;
  }

  try {
    setAuthError("");
    if (authMode === "signup") {
      await auth.createUserWithEmailAndPassword(emailValue, passwordValue);
    } else {
      await auth.signInWithEmailAndPassword(emailValue, passwordValue);
    }
    localStorage.setItem(AUTH_PROMPT_SEEN_KEY, "1");
    await syncAuthTokens();
    closeAuthModal(true);
  } catch (error) {
    setAuthError(error?.message || "Unable to sign in right now.");
  }
}

async function handleGoogleSignIn() {
  const auth = getAuthInstance();
  if (!auth) return;

  try {
    const provider = new window.firebase.auth.GoogleAuthProvider();
    await auth.signInWithPopup(provider);
    localStorage.setItem(AUTH_PROMPT_SEEN_KEY, "1");
    await syncAuthTokens();
    closeAuthModal(true);
  } catch (error) {
    setAuthError(error?.message || "Google sign-in failed.");
  }
}

async function handleLogout() {
  const auth = getAuthInstance();
  if (!auth) return;
  await auth.signOut();
  await syncAuthTokens();
  updateAuthButtons(null);
  openAuthModal({
    blocking: localStorage.getItem(AUTH_PROMPT_SEEN_KEY) === "1",
  });
}

async function getAuthHeaders(extraHeaders = {}) {
  const token = await syncAuthTokens();
  if (!token) {
    return { ...extraHeaders };
  }
  return {
    ...extraHeaders,
    Authorization: `Bearer ${token}`,
  };
}

async function requireAuthForAction(options = {}) {
  const auth = getAuthInstance();
  const user = auth?.currentUser || null;
  if (user) {
    return true;
  }

  openAuthModal({
    blocking: true,
    reason: options.reason || "Please sign in to continue.",
  });
  return false;
}

function maybeShowInitialAuthPrompt(user) {
  if (user) {
    closeAuthModal(true);
    return;
  }

  const alreadySeen = localStorage.getItem(AUTH_PROMPT_SEEN_KEY) === "1";
  if (!alreadySeen) {
    localStorage.setItem(AUTH_PROMPT_SEEN_KEY, "1");
    openAuthModal({ blocking: false });
    return;
  }

  openAuthModal({ blocking: true });
}

function wireAuthUi() {
  const {
    closeBtn,
    backdrop,
    form,
    googleBtn,
    signinTab,
    signupTab,
  } = getAuthModalElements();
  const enabledProviders = window.CROPGUARD_FIREBASE_PROVIDERS || [];

  document.getElementById("auth-open-desktop")?.addEventListener("click", () => {
    openAuthModal({ blocking: false });
  });
  document.getElementById("auth-open-mobile")?.addEventListener("click", () => {
    openAuthModal({ blocking: false });
  });
  document.getElementById("auth-logout-desktop")?.addEventListener("click", handleLogout);
  document.getElementById("auth-logout-mobile")?.addEventListener("click", handleLogout);

  closeBtn?.addEventListener("click", () => closeAuthModal());
  backdrop?.addEventListener("click", () => closeAuthModal());
  form?.addEventListener("submit", handleEmailAuth);
  googleBtn?.addEventListener("click", handleGoogleSignIn);
  signinTab?.addEventListener("click", () => setAuthMode("signin"));
  signupTab?.addEventListener("click", () => setAuthMode("signup"));

  if (googleBtn && !enabledProviders.includes("google")) {
    googleBtn.hidden = true;
  }
}

function wireProtectedForms() {
  document.querySelectorAll('form[action="/upload"]').forEach((form) => {
    form.dataset.requiresAuth = "true";
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await submitProtectedForm(form);
    });
  });
}

function startAuthStateListener() {
  const auth = getAuthInstance();
  if (!auth) return;

  auth.onAuthStateChanged(async (user) => {
    updateAuthButtons(user);
    await syncAuthTokens();
    maybeShowInitialAuthPrompt(user);
  });
}

window.getAuthHeaders = getAuthHeaders;
window.requireAuthForAction = requireAuthForAction;
window.submitProtectedForm = submitProtectedForm;
window.syncAuthTokens = syncAuthTokens;

document.addEventListener("DOMContentLoaded", () => {
  if (!initFirebaseAuth()) {
    return;
  }

  wireAuthUi();
  wireProtectedForms();
  setAuthMode("signin");
  startAuthStateListener();
});
