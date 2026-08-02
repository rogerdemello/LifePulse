// main.js - LifePulse Enhanced Global JavaScript

document.addEventListener('DOMContentLoaded', () => {
  console.log("LifePulse loaded 🚀");

  // ========================================
  // SMOOTH SCROLL
  // ========================================
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      // A bare "#" is not a valid selector; querySelector throws on it.
      if (!href || href.length < 2) return;
      const target = document.querySelector(href);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });

      // preventDefault() cancels the fragment navigation, and with it the
      // focus move the browser would have made. That is what broke the skip
      // link on every page: it scrolled, focus stayed on the link, and the
      // next Tab went to the navbar brand -- back into the navigation the
      // link exists to skip. Scrolling a sighted user to the content while
      // leaving a keyboard user where they were is the one thing it must not
      // do.
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });
    });
  });

  // ========================================
  // LOADING OVERLAY
  // ========================================
  const forms = document.querySelectorAll('form[method="post"]');
  const loadingOverlay = document.getElementById('loadingOverlay');
  
  forms.forEach(form => {
    form.addEventListener('submit', function(e) {
      // Check if form is valid before showing loading
      if (form.checkValidity()) {
        if (loadingOverlay) {
          loadingOverlay.classList.add('active');
          // Add subtle animation
          setTimeout(() => {
            const spinner = loadingOverlay.querySelector('.spinner-border');
            if (spinner) {
              spinner.style.transform = 'scale(1.1)';
              setTimeout(() => {
                spinner.style.transform = 'scale(1)';
              }, 200);
            }
          }, 100);
        }
      }
    });
  });

  // ========================================
  // FLASH MESSAGES
  // ========================================
  const flash = document.querySelector('.alert.alert-dismissible, .flash-message');
  if (flash && !flash.classList.contains('alert-warning') && !flash.classList.contains('alert-danger')) {
    setTimeout(() => {
      flash.style.opacity = '0';
      flash.style.transform = 'translateY(-20px)';
      setTimeout(() => {
        flash.remove();
      }, 300);
    }, 4000);
  }

  // ========================================
  // ENHANCED CARD INTERACTIONS
  // ========================================
  const cards = document.querySelectorAll('.hover-card');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function(e) {
      this.style.transform = 'translateY(-12px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function(e) {
      this.style.transform = 'translateY(0) scale(1)';
    });
  });

  // ========================================
  // FORM INPUT ENHANCEMENTS
  // ========================================
  const formInputs = document.querySelectorAll('.form-control, .form-select');
  
  formInputs.forEach(input => {
    // Add floating label effect
    input.addEventListener('focus', function() {
      this.parentElement.classList.add('focused');
    });
    
    input.addEventListener('blur', function() {
      if (!this.value) {
        this.parentElement.classList.remove('focused');
      }
    });
    
    // Mark the field, but let the browser show its own message.
    //
    // This used to call preventDefault(), which suppressed native validation
    // entirely -- no message, no focus jump to the offending field -- and
    // replaced it with a toast built from previousElementSibling.textContent.
    // For the blood-pressure input group that sibling is the "/" separator, so
    // the toast read "/ is required". Native validation is announced by screen
    // readers and moves focus correctly; there is nothing to improve on here.
    input.addEventListener('invalid', function() {
      this.classList.add('is-invalid');
    });


    input.addEventListener('input', function() {
      if (this.classList.contains('is-invalid')) {
        this.classList.remove('is-invalid');
      }
    });
  });

  // Form submission success handler
  forms.forEach(form => {
    const originalAction = form.action;
    form.addEventListener('submit', function(e) {
      if (!form.checkValidity()) {
        e.preventDefault();
        if (window.toast) {
          toast.warning('Please fill in all required fields', 3000);
        }
        return false;
      }
    });
  });

  // ========================================
  // NAVBAR SCROLL EFFECT
  // ========================================
  const navbar = document.querySelector('.navbar');
  let lastScroll = 0;
  
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
      navbar.style.boxShadow = '0 6px 30px rgba(0, 0, 0, 0.12)';
      navbar.style.padding = '0.75rem 0';
    } else {
      navbar.style.boxShadow = '0 4px 30px rgba(0, 0, 0, 0.08)';
      navbar.style.padding = '1rem 0';
    }
    
    lastScroll = currentScroll;
  });

  // ========================================
  // PARALLAX EFFECT FOR HERO SECTION - DISABLED TO PREVENT OVERLAP
  // ========================================
  // const heroSection = document.querySelector('.hero-section');
  // if (heroSection) {
  //   window.addEventListener('scroll', throttle(() => {
  //     const scrolled = window.pageYOffset;
  //     if (scrolled < window.innerHeight) {
  //       const parallax = scrolled * 0.3;
  //       heroSection.style.transform = `translateY(${parallax}px)`;
  //     }
  //   }, 16));
  // }

  // ========================================
  // BUTTON RIPPLE EFFECT
  // ========================================
  const buttons = document.querySelectorAll('.btn');
  buttons.forEach(button => {
    button.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      const rect = this.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = x + 'px';
      ripple.style.top = y + 'px';
      ripple.classList.add('ripple');
      
      this.appendChild(ripple);
      
      setTimeout(() => {
        ripple.remove();
      }, 600);
    });
  });

  // ========================================
  // NUMBER INPUT SCROLL PREVENTION
  // Stops a scroll over a focused number field from silently changing a health
  // value. Only while focused, so scrolling the page still works normally.
  // ========================================
  document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('wheel', function(e) {
      if (document.activeElement === this) {
        e.preventDefault();
      }
    }, { passive: false });
  });

  // ========================================
  // PERFORMANCE OPTIMIZATION
  // ========================================
  // Lazy load images if any
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.classList.add('loaded');
          observer.unobserve(img);
        }
      });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
      imageObserver.observe(img);
    });
  }

  // ========================================
  // PAGE LOAD
  //
  // There used to be a fade-in here that set document.body.style.opacity = '0'
  // and restored it 100ms later. If a script failed, was blocked, or simply
  // hadn't run yet, the page stayed permanently blank -- a health tool that
  // renders nothing at all rather than degrading. The CSS rule that hid the
  // body by default is gone too; content is visible without JavaScript.
  // ========================================
});

// ========================================
// LOADING OVERLAY -- BACK BUTTON
//
// Returning via the back button restores the page from the bfcache with the
// overlay still showing, leaving the user staring at "Analyzing Your Data..."
// over a page that finished loading long ago.
// ========================================
window.addEventListener('pageshow', () => {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.remove('active');
});

// ========================================
// UTILITY FUNCTIONS
// ========================================

// Debounce function for performance
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Throttle function for scroll events
function throttle(func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}
