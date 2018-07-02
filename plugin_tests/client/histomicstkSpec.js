/* globals girder, girderTest, describe, it, expect, waitsFor, runs */

girderTest.addScripts([
    '/clients/web/static/built/plugins/jobs/plugin.min.js',
    '/clients/web/static/built/plugins/worker/plugin.min.js',
    '/clients/web/static/built/plugins/large_image/plugin.min.js',
    '/clients/web/static/built/plugins/slicer_cli_web/plugin.min.js',
    '/clients/web/static/built/plugins/HistomicsTK/plugin.min.js'
]);

girderTest.startApp();

describe('Test the HistomicsTK plugin', function () {
    it('change the HistomicsTK settings', function () {
        var styles = [{'lineWidth': 8, 'id': 'Sample Group'}];
        var styleJSON = JSON.stringify(styles);

        girderTest.login('admin', 'Admin', 'Admin', 'password')();
        waitsFor(function () {
            return $('a.g-nav-link[g-target="admin"]').length > 0;
        }, 'admin console link to load');
        runs(function () {
            $('a.g-nav-link[g-target="admin"]').click();
        });
        waitsFor(function () {
            return $('.g-plugins-config').length > 0;
        }, 'the admin console to load');
        runs(function () {
            $('.g-plugins-config').click();
        });
        waitsFor(function () {
            return $('input.g-plugin-switch[key="HistomicsTK"]').length > 0;
        }, 'the plugins page to load');
        girderTest.waitForLoad();
        runs(function () {
            expect($('.g-plugin-config-link[g-route="plugins/HistomicsTK/config"]').length > 0);
            $('.g-plugin-config-link[g-route="plugins/HistomicsTK/config"]').click();
        });
        waitsFor(function () {
            return $('#g-histomicstk-form input').length > 0;
        }, 'settings to be shown');
        girderTest.waitForLoad();
        runs(function () {
            $('#g-histomicstk-default-draw-styles').val(styleJSON);
            $('.g-histomicstk-buttons .btn-primary').click();
        });
        waitsFor(function () {
            var resp = girder.rest.restRequest({
                url: 'system/setting',
                method: 'GET',
                data: {
                    list: JSON.stringify([
                        'histomicstk.default_draw_styles'
                    ])
                },
                async: false
            });
            var settings = resp.responseJSON;
            var settingsStyles = settings && JSON.parse(settings['histomicstk.default_draw_styles']);
            return (settingsStyles && settingsStyles.length === 1 &&
                    settingsStyles[0].lineWidth === styles[0].lineWidth);
        }, 'HistomicsTK settings to change');
        girderTest.waitForLoad();
        runs(function () {
            $('#g-histomicstk-default-draw-styles').val('not a json list');
            $('.g-histomicstk-buttons .btn-primary').click();
        });
        waitsFor(function () {
            return $('#g-histomicstk-error-message').text().substr('must be a JSON list') >= 0;
        });
        runs(function () {
            $('#g-histomicstk-brand-color').val('#ffffff');
            $('#g-histomicstk-brand-default-color').click();
            expect($('#g-histomicstk-brand-color').val() === '#777777');
            $('#g-histomicstk-banner-color').val('#ffffff');
            $('#g-histomicstk-banner-default-color').click();
            expect($('#g-histomicstk-banner-color').val() === '#f8f8f8');
        });
        runs(function () {
            $('.g-histomicstk-buttons #g-histomicstk-cancel').click();
        });
        waitsFor(function () {
            return $('input.g-plugin-switch[key="HistomicsTK"]').length > 0;
        }, 'the plugins page to load');
        girderTest.waitForLoad();
    });
});
