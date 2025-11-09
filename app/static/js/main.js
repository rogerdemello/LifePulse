// main.js - LifePulse Enhanced Global JavaScript

document.addEventListener('DOMContentLoaded', () => {
  console.log("LifePulse loaded 🚀");

  // ========================================
  // SMOOTH SCROLL
  // ========================================
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
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
    
    // Add validation feedback with toast
    input.addEventListener('invalid', function(e) {
      e.preventDefault();
      this.classList.add('is-invalid');
      this.style.borderColor = '#f5576c';
      
      // Show toast notification
      if (window.toast) {
        const label = this.previousElementSibling ? this.previousElementSibling.textContent : 'This field';
        toast.error(`${label} is required`, 3000);
      }
    });
    
    input.addEventListener('input', function() {
      if (this.classList.contains('is-invalid')) {
        this.classList.remove('is-invalid');
        this.style.borderColor = '';
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
  // ========================================
  const numberInputs = document.querySelectorAll('input[type="number"]');
  numberInputs.forEach(input => {
    input.addEventListener('wheel', function(e) {
      e.preventDefault();
    });
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
  // PAGE LOAD ANIMATION
  // ========================================
  document.body.style.opacity = '0';
  setTimeout(() => {
    document.body.style.transition = 'opacity 0.5s ease';
    document.body.style.opacity = '1';
  }, 100);

  console.log("✨ All enhancements loaded successfully!");
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
