/* Visit summary: assembled and stored entirely in the browser.
 *
 * Assessment results never come back to the server after they are rendered.
 * localStorage is same-origin and is not attached to requests the way a cookie
 * would be, so "nothing is stored" stays literally true server-side even when
 * the user chooses to keep a result.
 */
(function () {
  'use strict';

  var KEY = 'lifepulse.visitSummary';
  var LIMIT = 12;

  function read() {
    try {
      var raw = window.localStorage.getItem(KEY);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      // Private browsing, disabled storage, or corrupted JSON. Saving is a
      // convenience -- never let it break the page.
      return [];
    }
  }

  function write(entries) {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(entries.slice(-LIMIT)));
      return true;
    } catch (err) {
      return false;
    }
  }

  // ---- saving, on a result page -----------------------------------------

  function wireSaveButton() {
    var payloadEl = document.getElementById('assessmentSummary');
    var button = document.querySelector('[data-save-summary]');
    if (!payloadEl || !button) return;

    var payload;
    try {
      payload = JSON.parse(payloadEl.textContent);
    } catch (err) {
      button.disabled = true;
      return;
    }

    button.addEventListener('click', function () {
      var entries = read().filter(function (entry) {
        return entry.title !== payload.title;  // one entry per assessment type
      });
      entries.push(payload);

      var confirmation = document.querySelector('[data-save-confirmation]');
      if (write(entries)) {
        if (confirmation) confirmation.classList.remove('d-none');
        button.innerHTML = '<i class="bi bi-check-lg me-1"></i>Saved';
        button.classList.replace('btn-primary', 'btn-success');
        if (window.toast) window.toast.success('Added to your visit summary', 3000);
      } else if (window.toast) {
        window.toast.error('Your browser is blocking local storage, so this could not be saved', 5000);
      }
    });
  }

  // ---- rendering, on /summary -------------------------------------------

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderEntry(entry) {
    var card = el('section', 'summary-entry');

    var head = el('div', 'summary-entry-head');
    head.appendChild(el('h2', 'h5 fw-bold mb-1', entry.title));
    head.appendChild(el('p', 'text-muted small mb-0', 'Completed ' + entry.date));
    card.appendChild(head);

    var headline = el('p', 'summary-headline', entry.headline);
    card.appendChild(headline);
    if (entry.detail) card.appendChild(el('p', 'text-muted mb-3', entry.detail));

    (entry.flags || []).forEach(function (flag) {
      var box = el('div', 'summary-flag' + (flag.urgency === 'emergency' ? ' summary-flag-emergency' : ''));
      box.appendChild(el('strong', null, flag.title));
      box.appendChild(el('p', 'mb-0 small', flag.detail));
      card.appendChild(box);
    });

    (entry.caveats || []).forEach(function (text) {
      card.appendChild(el('p', 'summary-caveat small', text));
    });

    if ((entry.factors || []).length) {
      card.appendChild(el('h3', 'h6 fw-bold mt-3 mb-2', 'What drove this'));
      var list = el('ul', 'summary-factors');
      entry.factors.forEach(function (factor) {
        var item = el('li');
        item.appendChild(el('span', 'summary-arrow ' + factor.direction,
          factor.direction === 'raised' ? '↑' : '↓'));
        item.appendChild(document.createTextNode(
          ' ' + factor.label + ' (' + factor.value + ') ' +
          factor.direction + ' it by ' + factor.delta
        ));
        list.appendChild(item);
      });
      card.appendChild(list);
    }

    if ((entry.inputs || []).length) {
      var details = el('details', 'summary-inputs');
      details.appendChild(el('summary', null, 'What I entered'));
      var dl = el('dl', 'row small mb-0 mt-2');
      entry.inputs.forEach(function (input) {
        dl.appendChild(el('dt', 'col-6 col-sm-4 fw-normal text-muted', input.label));
        dl.appendChild(el('dd', 'col-6 col-sm-8', input.value));
      });
      details.appendChild(dl);
      card.appendChild(details);
    }

    return card;
  }

  function renderSummary() {
    var root = document.getElementById('summaryRoot');
    if (!root) return;

    var entries = read();
    var empty = document.getElementById('summaryEmpty');
    var content = document.getElementById('summaryContent');
    var questionsCard = document.getElementById('summaryQuestions');

    if (!entries.length) {
      if (empty) empty.classList.remove('d-none');
      return;
    }
    if (content) content.classList.remove('d-none');

    entries.forEach(function (entry) {
      root.appendChild(renderEntry(entry));
    });

    // Questions across every saved assessment, de-duplicated. Emergencies
    // first, because those are the ones that must not get lost in a list.
    var seen = {};
    var questions = [];
    entries.forEach(function (entry) {
      (entry.questions || []).forEach(function (question) {
        if (!seen[question]) {
          seen[question] = true;
          questions.push(question);
        }
      });
    });

    if (questions.length && questionsCard) {
      questionsCard.classList.remove('d-none');
      var list = document.getElementById('summaryQuestionList');
      questions.forEach(function (question) {
        var item = el('li', 'mb-2');
        item.appendChild(el('span', 'summary-checkbox', '☐'));
        item.appendChild(document.createTextNode(' ' + question));
        list.appendChild(item);
      });
    }

    var printBtn = document.getElementById('summaryPrint');
    if (printBtn) printBtn.addEventListener('click', function () { window.print(); });

    var clearBtn = document.getElementById('summaryClear');
    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        if (!window.confirm('Delete every saved result from this browser?')) return;
        try { window.localStorage.removeItem(KEY); } catch (err) { /* nothing to do */ }
        window.location.reload();
      });
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    wireSaveButton();
    renderSummary();
  });
})();
