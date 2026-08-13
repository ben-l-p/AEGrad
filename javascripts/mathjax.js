/* global MathJax, document$ */
window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

document$.subscribe(() => {
    // noinspection JSUnresolvedReference
    MathJax.startup.output.clearCache();
    // noinspection JSUnresolvedReference
    MathJax.typesetClear();
    // noinspection JSUnresolvedReference
    MathJax.texReset();
    // noinspection JSUnresolvedReference
    MathJax.typesetPromise();
});
