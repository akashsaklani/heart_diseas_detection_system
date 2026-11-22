'use strict';

const form = document.getElementById('prediction-form');
const predictButton = document.getElementById('predict-button');
const emptyResult = document.getElementById('result-empty');
const errorResult = document.getElementById('result-error');
const resultContent = document.getElementById('result-content');
const patientPane = document.getElementById('patient-view-content');
const technicalPane = document.getElementById('technical-view-content');
const patientButton = document.getElementById('patient-view-button');
const technicalButton = document.getElementById('technical-view-button');
const ringCircumference = 2 * Math.PI * 48;
const viewStorageKey = 'cardiocheck.resultView';

const sampleValues = {
  age: 45,
  sex: 1,
  cp: 0,
  trestbps: 130,
  chol: 230,
  fbs: 0,
  restecg: 1,
  thalach: 150,
  exang: 0,
  oldpeak: 1.0,
  slope: 1,
};

let currentView = 'patient';
try {
  const savedView = localStorage.getItem(viewStorageKey);
  if (savedView === 'technical') currentView = savedView;
} catch {
  // Keep the default view when browser storage is unavailable.
}

function setView(view, remember = false) {
  currentView = view === 'technical' ? 'technical' : 'patient';
  const isPatientView = currentView === 'patient';

  patientButton.setAttribute('aria-pressed', String(isPatientView));
  technicalButton.setAttribute('aria-pressed', String(!isPatientView));
  patientButton.classList.toggle('is-selected', isPatientView);
  technicalButton.classList.toggle('is-selected', !isPatientView);

  patientPane.classList.toggle('is-active', isPatientView);
  technicalPane.classList.toggle('is-active', !isPatientView);
  patientPane.setAttribute('aria-hidden', String(!isPatientView));
  technicalPane.setAttribute('aria-hidden', String(isPatientView));
  patientPane.inert = !isPatientView;
  technicalPane.inert = isPatientView;

  if (remember) {
    try {
      localStorage.setItem(viewStorageKey, currentView);
    } catch {
      // The view still changes for this page when browser storage is unavailable.
    }
  }
}

function linkedNumber(fieldName) {
  return form.querySelector(`[data-field="${fieldName}"]`);
}

function updateRangeDisplay(range) {
  const number = linkedNumber(range.name);
  const output = document.getElementById(`${range.name}-value`);
  if (number) number.value = range.value;
  if (output) output.value = range.name === 'oldpeak' ? Number(range.value).toFixed(1) : range.value;
}

function fieldMessage(control) {
  if (control instanceof HTMLSelectElement) {
    return control.value ? '' : 'Please choose an option.';
  }

  if (control.validity.badInput || !control.value) return 'Please provide a valid value.';
  const value = Number(control.value);
  const min = Number(control.min);
  const max = Number(control.max);

  if (!Number.isFinite(value)) return 'Please enter a valid number.';
  if (value < min || value > max) return `Choose a value between ${control.min} and ${control.max}.`;
  if (control.validity.stepMismatch) return `Use increments of ${control.step}.`;
  return '';
}

function validateField(control) {
  const fieldName = control.name || control.dataset.field;
  const message = fieldMessage(control);
  const error = document.getElementById(`${fieldName}-error`);
  control.setAttribute('aria-invalid', message ? 'true' : 'false');
  if (error) error.textContent = message;

  const range = form.elements.namedItem(fieldName);
  if (range instanceof HTMLInputElement && range.type === 'range') {
    range.setAttribute('aria-invalid', message ? 'true' : 'false');
  }
  return !message;
}

function validateForm() {
  const controls = Array.from(form.querySelectorAll('input[type="number"][data-field], select[required]'));
  const validity = controls.map(validateField);
  const firstInvalid = controls[validity.indexOf(false)];
  if (firstInvalid) firstInvalid.focus();
  return validity.every(Boolean);
}

function showEmptyState() {
  emptyResult.hidden = false;
  errorResult.hidden = true;
  resultContent.hidden = true;
}

function showError(title, message, detail) {
  document.getElementById('error-title').textContent = title;
  document.getElementById('error-message').textContent = message;
  const detailElement = document.getElementById('error-detail');
  detailElement.textContent = detail || '';
  detailElement.hidden = !detail;
  emptyResult.hidden = true;
  resultContent.hidden = true;
  errorResult.hidden = false;
}

function selectedText(fieldName) {
  const select = form.elements.namedItem(fieldName);
  return select.options[select.selectedIndex].textContent.trim().replace(/\s*\([^)]*\)\s*$/, '');
}

function populatePatientDetails() {
  const values = {
    age: `${form.elements.namedItem('age').value} years`,
    sex: form.elements.namedItem('sex').value === '1' ? 'Male' : 'Female',
    cp: selectedText('cp'),
    trestbps: `${form.elements.namedItem('trestbps').value} mmHg`,
    chol: `${form.elements.namedItem('chol').value} mg/dl`,
    fbs: form.elements.namedItem('fbs').value === '1' ? 'Yes' : 'No',
    restecg: selectedText('restecg'),
    thalach: `${form.elements.namedItem('thalach').value} bpm`,
    exang: form.elements.namedItem('exang').value === '1' ? 'Yes' : 'No',
    oldpeak: form.elements.namedItem('oldpeak').value,
    slope: selectedText('slope'),
  };

  Object.entries(values).forEach(([name, value]) => {
    document.getElementById(`detail-${name}`).textContent = value;
  });
}

function showPrediction(data) {
  const prediction = Number(data.prediction);
  const probabilityZero = Number(data.probability_class_0);
  const probabilityOne = Number(data.probability_class_1);
  if (![0, 1].includes(prediction)
      || !Number.isFinite(probabilityZero)
      || !Number.isFinite(probabilityOne)
      || probabilityZero < 0 || probabilityZero > 1
      || probabilityOne < 0 || probabilityOne > 1) {
    showError(
      'We couldn’t read this estimate',
      'The prediction service returned an unexpected response. Please try again.',
    );
    return;
  }

  const outcome = document.getElementById('patient-outcome');
  const resultLabel = document.getElementById('patient-result-label');
  const headline = document.getElementById('patient-headline');
  const chance = document.getElementById('chance-meter');
  const marker = document.getElementById('meter-marker');
  let level;
  let wording;
  let headlineText;

  if (probabilityOne < 0.35) {
    level = 'low';
    wording = 'lower';
    headlineText = 'Lower chance of heart problems';
    resultLabel.textContent = 'Lower chance';
  } else if (probabilityOne <= 0.65) {
    level = 'moderate';
    wording = 'moderate';
    headlineText = 'Moderate chance, worth a check-up';
    resultLabel.textContent = 'Moderate chance';
  } else {
    level = 'high';
    wording = 'higher';
    headlineText = 'Higher chance of heart problems, please see a doctor';
    resultLabel.textContent = 'Higher chance';
  }

  outcome.classList.remove('is-low', 'is-moderate', 'is-high');
  outcome.classList.add(`is-${level}`);
  headline.textContent = headlineText;
  document.getElementById('patient-summary').textContent =
    `Based on the details you entered, the system sees a ${wording} chance of heart problems. This is not a diagnosis.`;
  chance.setAttribute('aria-label', `Likelihood range: ${level === 'low' ? 'Low' : level === 'moderate' ? 'Moderate' : 'High'}`);
  marker.style.left = `${Math.max(1, Math.min(99, probabilityOne * 100))}%`;
  populatePatientDetails();

  const percentZero = probabilityZero * 100;
  const percentOne = probabilityOne * 100;
  const ring = document.querySelector('.probability-ring');
  const progress = document.getElementById('ring-progress');
  const badge = document.getElementById('prediction-badge');

  document.getElementById('probability-value').textContent = `${Math.round(percentOne)}%`;
  document.getElementById('prediction-label').textContent = `Class ${prediction}`;
  document.getElementById('class-0-value').textContent = `${percentZero.toFixed(1)}%`;
  document.getElementById('class-1-value').textContent = `${percentOne.toFixed(1)}%`;
  document.getElementById('class-0-bar').style.width = `${percentZero}%`;
  document.getElementById('class-1-bar').style.width = `${percentOne}%`;
  badge.textContent = prediction === 1 ? 'Class 1 · higher likelihood' : 'Class 0 · lower likelihood';
  badge.classList.toggle('is-class-one', prediction === 1);
  ring.setAttribute('aria-valuenow', percentOne.toFixed(1));
  progress.style.strokeDasharray = `${ringCircumference}`;
  progress.style.strokeDashoffset = `${ringCircumference}`;

  emptyResult.hidden = true;
  errorResult.hidden = true;
  resultContent.hidden = false;
  requestAnimationFrame(() => {
    progress.style.strokeDashoffset = `${ringCircumference * (1 - probabilityOne)}`;
  });
}

form.querySelectorAll('input[type="range"]').forEach((range) => {
  updateRangeDisplay(range);
  range.addEventListener('input', () => {
    updateRangeDisplay(range);
    validateField(linkedNumber(range.name));
  });
});

form.querySelectorAll('input[type="number"][data-field]').forEach((number) => {
  number.addEventListener('input', () => {
    const range = form.elements.namedItem(number.dataset.field);
    if (number.value && number.checkValidity()) {
      range.value = number.value;
      updateRangeDisplay(range);
    }
    validateField(number);
  });
});

form.querySelectorAll('select').forEach((select) => {
  select.addEventListener('change', () => validateField(select));
});

patientButton.addEventListener('click', () => setView('patient', true));
technicalButton.addEventListener('click', () => setView('technical', true));
setView(currentView);

document.getElementById('sample-button').addEventListener('click', () => {
  Object.entries(sampleValues).forEach(([name, value]) => {
    const control = form.elements.namedItem(name);
    control.value = value;
    control.dispatchEvent(new Event('input', { bubbles: true }));
    control.dispatchEvent(new Event('change', { bubbles: true }));
  });
  showEmptyState();
});

document.getElementById('reset-button').addEventListener('click', () => {
  window.setTimeout(() => {
    form.querySelectorAll('[aria-invalid]').forEach((control) => control.removeAttribute('aria-invalid'));
    form.querySelectorAll('.field-error').forEach((error) => { error.textContent = ''; });
    form.querySelectorAll('input[type="range"]').forEach(updateRangeDisplay);
    showEmptyState();
  }, 0);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!validateForm()) return;

  predictButton.disabled = true;
  predictButton.classList.add('is-loading');
  predictButton.querySelector('.button-label').textContent = 'Calculating estimate';

  try {
    const payload = Object.fromEntries(new FormData(form));
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = typeof data.error === 'string' ? data.error : '';
      if (response.status === 400) {
        showError(
          'Please check your details',
          'One or more values could not be accepted. Review the measurements and try again.',
          detail,
        );
      } else if (response.status === 503) {
        showError(
          'The estimate is temporarily unavailable',
          'The prediction model is not ready right now. Please try again later.',
          detail,
        );
      } else {
        showError(
          'We couldn’t make an estimate',
          'Something went wrong while preparing your result. Please try again.',
          detail,
        );
      }
      return;
    }

    showPrediction(data);
  } catch {
    showError(
      'Unable to reach CardioCheck',
      'Please check your connection and try the prediction again.',
    );
  } finally {
    predictButton.disabled = false;
    predictButton.classList.remove('is-loading');
    predictButton.querySelector('.button-label').textContent = 'Get my estimate';
  }
});
