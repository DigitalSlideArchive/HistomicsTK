// Add all necessary scripts into the testing environment.
(function () {
    _.each([
        '/plugins/HistomicsTK/web_client/js/ext/backbone.localStorage.js',
        '/plugins/HistomicsTK/web_client/js/ext/bootstrap-colorpicker.js',
        '/plugins/HistomicsTK/web_client/js/ext/bootstrap-slider.js',
        '/plugins/HistomicsTK/web_client/js/ext/tinycolor.js'
    ], function (src) {
        $('<script/>', {src: src}).appendTo('head');
    });

    window.histomicstk = {};
    girderTest.addCoveredScripts([
        '/plugins/HistomicsTK/web_client/js/0init.js',
        '/plugins/HistomicsTK/web_client/js/app.js',
        '/plugins/HistomicsTK/web_client/js/models/widget.js',
        '/plugins/HistomicsTK/web_client/js/schema/parser.js',
        '/plugins/HistomicsTK/web_client/js/views/0panel.js',
        '/plugins/HistomicsTK/web_client/js/views/body.js',
        '/plugins/HistomicsTK/web_client/js/views/browserPanel.js',
        '/plugins/HistomicsTK/web_client/js/views/controlsPanel.js',
        '/plugins/HistomicsTK/web_client/js/views/controlWidget.js',
        '/plugins/HistomicsTK/web_client/js/views/fileSelectorWidget.js',
        '/plugins/HistomicsTK/web_client/js/views/guiSelectorWidget.js',
        '/plugins/HistomicsTK/web_client/js/views/header.js',
        '/plugins/HistomicsTK/web_client/js/views/jobsPanel.js',
        '/plugins/HistomicsTK/web_client/js/views/panelGroup.js',
        '/plugins/HistomicsTK/web_client/js/views/visualization.js',
        '/clients/web/static/built/plugins/HistomicsTK/templates.js'
    ]);

    beforeEach(function () {
        waitsFor(function () {
             // the templates are loaded last, so wait until blanket finishes instrumenting them before
             // starting tests
             return !!(histomicstk && histomicstk.templates && histomicstk.templates.visualization);
        });
    });
})();
