// main.js — CardioPredict UI enhancements

document.addEventListener("DOMContentLoaded", () => {

  // ── Form validation feedback ──────────────────────────────────
  const form = document.getElementById("predictionForm");
  if (form) {
    form.addEventListener("submit", (e) => {
      const btn = form.querySelector(".btn-predict");
      btn.textContent = "Analyzing...";
      btn.disabled = true;
    });

    // Inline numeric range hints
    const rangeHints = {
      age: { min: 1, max: 120, msg: "Age should be between 1 and 120." },
      trestbps: { min: 80, max: 250, msg: "Blood pressure should be between 80–250 mm Hg." },
      chol: { min: 100, max: 600, msg: "Cholesterol should be between 100–600 mg/dl." },
      thalach: { min: 50, max: 250, msg: "Heart rate should be between 50–250." },
      oldpeak: { min: 0, max: 10, msg: "Oldpeak should be between 0 and 10." },
    };

    Object.entries(rangeHints).forEach(([id, { min, max, msg }]) => {
      const input = document.getElementById(id);
      if (!input) return;
      input.addEventListener("blur", () => {
        const val = parseFloat(input.value);
        let hint = input.parentElement.querySelector(".range-hint");
        if (!hint) {
          hint = document.createElement("span");
          hint.className = "range-hint";
          hint.style.cssText = "font-size:0.75rem;color:#c0392b;margin-top:2px;";
          input.parentElement.appendChild(hint);
        }
        if (input.value !== "" && (val < min || val > max)) {
          hint.textContent = msg;
          input.style.borderColor = "#e74c3c";
        } else {
          hint.textContent = "";
          input.style.borderColor = "";
        }
      });
    });
  }

  // ── Animate confidence bar on result page ────────────────────
  const fill = document.querySelector(".confidence-fill");
  if (fill) {
    const target = fill.style.width;
    fill.style.width = "0%";
    setTimeout(() => { fill.style.width = target; }, 200);
  }
});
