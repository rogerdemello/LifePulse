// Toast Notification System for LifePulse

class ToastNotification {
  constructor() {
    this.container = null;
    this.init();
  }

  init() {
    // Create toast container if it doesn't exist
    if (!document.getElementById('toast-container')) {
      this.container = document.createElement('div');
      this.container.id = 'toast-container';
      this.container.className = 'toast-container';
      // Without a live region a toast is invisible to a screen reader: the
      // message is appended to the DOM after load and nothing announces it.
      // Every result page fires one -- the heart page says "Assessment
      // complete. Please consult a healthcare provider." -- so the only
      // spoken confirmation that anything happened was never spoken.
      //
      // polite, not assertive: these follow an action the user just took, and
      // interrupting the result they are in the middle of hearing is worse
      // than waiting for a pause. aria-atomic="false" so a second toast does
      // not re-announce the first.
      this.container.setAttribute('role', 'status');
      this.container.setAttribute('aria-live', 'polite');
      this.container.setAttribute('aria-atomic', 'false');
      document.body.appendChild(this.container);
    } else {
      this.container = document.getElementById('toast-container');
    }
  }

  show(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Get icon based on type. aria-hidden because the icon repeats what the
    // message says, and an unlabelled icon font otherwise reads as nothing or
    // as a stray glyph.
    const icons = {
      success: '<i class="bi bi-check-circle-fill" aria-hidden="true"></i>',
      error: '<i class="bi bi-exclamation-circle-fill" aria-hidden="true"></i>',
      warning: '<i class="bi bi-exclamation-triangle-fill" aria-hidden="true"></i>',
      info: '<i class="bi bi-info-circle-fill" aria-hidden="true"></i>'
    };

    // The close button was a tabbable control whose entire content was an icon
    // font glyph, so it reached the accessibility tree unnamed -- a screen
    // reader announced "button" and nothing else. type="button" because a
    // button with no type is a submit button, and this one is appended to
    // document.body, which on a form page is not always outside the form.
    toast.innerHTML = `
      <div class="toast-icon">${icons[type]}</div>
      <div class="toast-message">${message}</div>
      <button type="button" class="toast-close" aria-label="Dismiss notification">
        <i class="bi bi-x" aria-hidden="true"></i>
      </button>
    `;
    toast.querySelector('.toast-close')
         .addEventListener('click', () => this.hide(toast));

    this.container.appendChild(toast);

    // Trigger animation
    setTimeout(() => {
      toast.classList.add('show');
    }, 10);

    // Auto remove, unless the user has tabbed into it. Pulling the focused
    // element out of the document sends focus to <body>, which costs a
    // keyboard user their place in the page -- so wait until they leave.
    if (duration > 0) {
      const dismiss = () => {
        if (toast.contains(document.activeElement)) {
          setTimeout(dismiss, 1000);
          return;
        }
        this.hide(toast);
      };
      setTimeout(dismiss, duration);
    }

    return toast;
  }

  hide(toast) {
    toast.classList.remove('show');
    toast.classList.add('hide');
    setTimeout(() => {
      toast.remove();
    }, 300);
  }

  success(message, duration) {
    return this.show(message, 'success', duration);
  }

  error(message, duration) {
    return this.show(message, 'error', duration);
  }

  warning(message, duration) {
    return this.show(message, 'warning', duration);
  }

  info(message, duration) {
    return this.show(message, 'info', duration);
  }
}

// Initialize global toast instance
const toast = new ToastNotification();

// Export for use in other scripts
if (typeof window !== 'undefined') {
  window.toast = toast;
}
