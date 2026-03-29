/* VeriAI - حركة بسيطة للشعار */

document.addEventListener('DOMContentLoaded', function() {
    const logo = document.getElementById('logoImg');
    if (logo) {
        logo.style.opacity = '0.9';
        logo.addEventListener('mouseenter', function() {
            logo.style.transform = 'scale(1.08)';
            logo.style.transition = 'transform 0.3s ease';
        });
        logo.addEventListener('mouseleave', function() {
            logo.style.transform = 'scale(1)';
        });
    }
});
