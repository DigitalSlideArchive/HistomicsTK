/* global histomicsTest */

girderTest.importPlugin('jobs');
girderTest.importPlugin('worker');
girderTest.importPlugin('large_image');
girderTest.importPlugin('slicer_cli_web');
girderTest.importPlugin('HistomicsTK');
girderTest.addScript('/plugins/HistomicsTK/plugin_tests/client/common.js');

var app;

girderTest.promise.done(function () {
    app = histomicsTest.startApp();
});

$(function () {
    var restRequest;
    beforeEach(function () {
        // Replace girder's rest request method with one that calls
        // our mocked docker_image endpoint in place of the real one.
        restRequest = girder.rest.restRequest;
        girder.rest.restRequest = function (opts) {
            if (opts.url === 'HistomicsTK/HistomicsTK/docker_image') {
                opts.url = 'mock_resource/docker_image';
            }
            return restRequest.call(this, opts);
        };
    });
    afterEach(function () {
        girder.rest.restRequest = restRequest;
    });
    describe('setup', function () {
        it('login', function () {
            histomicsTest.login();
        });

        it('open image', function () {
            histomicsTest.openImage('image');
        });
    });

    describe('open analysis', function () {
        it('get dropdown values', function () {
            var $el = $('.h-analyses-dropdown');
            expect($el.find('a:contains("dsarchive/histomicstk")').length).toBe(1);

            $el = $el.find('.dropdown-submenu:first');
            expect($el.find('a:contains("latest")').length).toBe(1);

            $el = $el.find('.dropdown-menu:first');
            var link = $el.find('a:contains("NucleiDetection")');
            expect(link.length).toBe(1);

            link.click();
            girderTest.waitForLoad();

            runs(function () {
                var $panel = $('.h-control-panel-container .s-panel:first');
                expect($panel.find('.s-panel-title-container').text()).toBe('Detects Nuclei');
            });
        });
        it('check autofilled forms', function () {
            waitsFor(function () {
                return !!$('#inputImageFile').val();
            }, 'Input image to auto fill');
            runs(function () {
                expect($('#inputImageFile').val()).toBe('image');
            });

            waitsFor(function () {
                return !!$('#outputNucleiAnnotationFile');
            }, 'Output annotation file to auto fill');

            runs(function () {
                expect($('#outputNucleiAnnotationFile').val()).toBe('test_analysis-outputNucleiAnnotationFile.anot');
            });
        });
        it('draw a region of interest', function () {
            var regionButton = $('.s-select-region-button');
            var interactor = histomicsTest.geojsMap().interactor();

            expect(regionButton.length).toBe(1);
            regionButton.click();

            interactor.simulateEvent('mousedown', {
                map: {x: 100, y: 100},
                button: 'left'
            });
            interactor.simulateEvent('mousemove', {
                map: {x: 200, y: 200},
                button: 'left'
            });
            interactor.simulateEvent('mouseup', {
                map: {x: 200, y: 200},
                button: 'left'
            });

            waitsFor(function () {
                return $('#analysis_roi').val() !== '-1,-1,-1,-1';
            }, 'roi widget to update');
        });
        it('assert roi resets on analysis change', function () {
            var resetCalled;
            app.bodyView.viewerWidget.on('g:mouseResetAnnotation', function (annotation) {
                if (annotation.id === 'region-selection') {
                    resetCalled = true;
                }
            });
            girder.plugins.HistomicsTK.events.trigger('h:analysis', null);
            waitsFor(function () {
                return resetCalled;
            }, 'region annotation to be removed');
        });
    });
});
