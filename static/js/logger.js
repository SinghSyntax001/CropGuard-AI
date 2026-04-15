(function () {
  if (window.CropGuardLogger) {
    return;
  }

  function emit(level, args) {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
    const method = console[level] || console.log;
    method.call(console, prefix, ...args);
  }

  window.CropGuardLogger = {
    debug: (...args) => emit("debug", args),
    info: (...args) => emit("info", args),
    warn: (...args) => emit("warn", args),
    error: (...args) => emit("error", args),
  };
})();