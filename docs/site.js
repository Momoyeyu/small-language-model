(() => {
  const links = [...document.querySelectorAll('.sidebar nav a[href^="#"]')];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);
  if (!links.length || !sections.length || !('IntersectionObserver' in window)) return;

  const activate = (id) => {
    links.forEach((link) => {
      link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
    });
  };
  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (visible) activate(visible.target.id);
  }, { rootMargin: '-20% 0px -65% 0px', threshold: [0, 0.2, 0.6] });
  sections.forEach((section) => observer.observe(section));
})();
