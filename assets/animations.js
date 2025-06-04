// Hover animation for buttons using GSAP
gsap.from(".button", {
    scale: 0.8,
    opacity: 0,
    duration: 0.5,
    delay: 0.3,
    ease: "back.out(1.7)"
});

$(".button").hover(
    function() {
        gsap.to(this, { scale: 1.1, backgroundColor: "#ff0099", duration: 0.3 });
    },
    function() {
        gsap.to(this, { scale: 1, backgroundColor: "#a00000", duration: 0.3 });
    }
);

// Smoothly animate the loading spinner
function showLoading() {
    document.querySelector(".spinner").style.display = "block";
}

function hideLoading() {
    document.querySelector(".spinner").style.display = "none";
}
