/* Multi-step forms, as an enhancement over a form that already works.
 *
 * The heart assessment is seventeen questions. Presented as one wall it reads
 * as a chore, and on a phone it is a very long scroll before anything happens.
 *
 * Progressive enhancement, deliberately: the markup is a normal form with
 * sections, and every field is present and submittable with JavaScript off.
 * This script only *hides* sections it has grouped into steps, so a failure to
 * load leaves a working long form rather than a blank page. (There is a test
 * asserting the pages render without JS -- see tests/test_frontend.py.)
 */
(function () {
  'use strict';

  function build(form) {
    var steps = Array.prototype.slice.call(form.querySelectorAll('[data-step]'));
    if (steps.length < 2) return;

    var actions = form.querySelector('[data-step-actions]');
    if (!actions) return;

    var current = 0;

    // --- progress -----------------------------------------------------
    var progress = document.createElement('div');
    progress.className = 'form-steps-progress mb-4';
    progress.innerHTML =
      '<div class="d-flex justify-content-between align-items-baseline mb-2">' +
      '  <span class="fw-semibold" data-step-title></span>' +
      '  <span class="text-muted small" data-step-count></span>' +
      '</div>' +
      '<div class="form-steps-bar"><span></span></div>';
    form.insertBefore(progress, form.firstChild);

    var title = progress.querySelector('[data-step-title]');
    var count = progress.querySelector('[data-step-count]');
    var bar = progress.querySelector('.form-steps-bar > span');

    // --- navigation ---------------------------------------------------
    var nav = document.createElement('div');
    nav.className = 'd-flex gap-2 justify-content-between mt-4';
    nav.innerHTML =
      '<button type="button" class="btn btn-outline-secondary" data-step-back>' +
      '  <i class="bi bi-arrow-left me-1"></i>Back</button>' +
      '<button type="button" class="btn btn-primary ms-auto" data-step-next>' +
      '  Next<i class="bi bi-arrow-right ms-1"></i></button>';
    form.insertBefore(nav, actions);

    var back = nav.querySelector('[data-step-back]');
    var next = nav.querySelector('[data-step-next]');

    // A live region so a screen reader is told the step changed; without it
    // the page appears to do nothing when Next is pressed.
    var announcer = document.createElement('p');
    announcer.className = 'visually-hidden';
    announcer.setAttribute('role', 'status');
    announcer.setAttribute('aria-live', 'polite');
    form.appendChild(announcer);

    function show(index, announce) {
      current = Math.max(0, Math.min(steps.length - 1, index));
      steps.forEach(function (step, i) {
        step.hidden = i !== current;
      });

      var label = steps[current].getAttribute('data-step') || ('Step ' + (current + 1));
      title.textContent = label;
      count.textContent = 'Step ' + (current + 1) + ' of ' + steps.length;
      bar.style.width = ((current + 1) / steps.length * 100) + '%';

      back.hidden = current === 0;
      var last = current === steps.length - 1;
      next.hidden = last;
      actions.hidden = !last;

      if (announce) {
        announcer.textContent = label + ', step ' + (current + 1) + ' of ' + steps.length;
        var focusable = steps[current].querySelector('input, select, textarea');
        if (focusable) focusable.focus({ preventScroll: true });
        progress.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }

    function stepIsValid(index) {
      var fields = steps[index].querySelectorAll('input, select, textarea');
      for (var i = 0; i < fields.length; i++) {
        if (!fields[i].checkValidity()) {
          // Let the browser do the complaining -- it announces properly and
          // moves focus to the offending field.
          fields[i].reportValidity();
          return false;
        }
      }
      return true;
    }

    next.addEventListener('click', function () {
      if (stepIsValid(current)) show(current + 1, true);
    });

    back.addEventListener('click', function () {
      show(current - 1, true);
    });

    // Enter should advance rather than submit a half-filled form.
    form.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter') return;
      if (event.target.tagName === 'TEXTAREA') return;
      if (current < steps.length - 1) {
        event.preventDefault();
        if (stepIsValid(current)) show(current + 1, true);
      }
    });

    // If the browser blocks submission for a field on an earlier step, that
    // step is hidden and the message would have nowhere to appear.
    form.addEventListener('invalid', function (event) {
      var owner = event.target.closest('[data-step]');
      if (!owner) return;
      var index = steps.indexOf(owner);
      if (index !== -1 && index !== current) show(index, true);
    }, true);

    show(0, false);
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('form[data-steps]').forEach(build);
  });
})();
