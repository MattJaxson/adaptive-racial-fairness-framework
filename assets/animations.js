document.addEventListener("DOMContentLoaded", function () {
    const darkToggle = document.getElementById("dark-mode-toggle");
    const contrastToggle = document.getElementById("contrast-toggle");

    if (darkToggle) {
        darkToggle.addEventListener("click", () => {
            document.body.classList.toggle("dark-mode");
        });
    }

    if (contrastToggle) {
        contrastToggle.addEventListener("click", () => {
            document.body.classList.toggle("high-contrast");
        });
    }

    // GSAP Button Animations
    gsap.from(".button", {
        scale: 0.8,
        opacity: 0,
        duration: 0.5,
        delay: 0.3,
        ease: "back.out(1.7)"
    });

    $(".button").hover(
        function () {
            gsap.to(this, { scale: 1.1, backgroundColor: "#ff0099", duration: 0.3 });
        },
        function () {
            gsap.to(this, { scale: 1, backgroundColor: "#a00000", duration: 0.3 });
        }
    );

    // Animate cards
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = "translateY(0) scale(1.02)";
            }
        });
    }, {
        threshold: 0.1
    });

    document.querySelectorAll(".card").forEach(card => {
        card.style.opacity = 0;
        card.style.transform = "translateY(20px) scale(0.98)";
        card.style.transition = "opacity 0.5s ease, transform 0.5s ease";
        observer.observe(card);
    });
});
