/* eslint-disable camelcase */

girderTest.importStylesheet(
    '/static/built/plugins/jobs/plugin.min.css'
);
girderTest.importStylesheet(
    '/static/built/plugins/slicer_cli_web/plugin.min.css'
);
girderTest.importStylesheet(
    '/static/built/plugins/large_image/plugin.min.css'
);
girderTest.importStylesheet(
    '/static/built/plugins/HistomicsTK/plugin.min.css'
);
girderTest.addCoveredScripts([
    '/clients/web/static/built/plugins/HistomicsTK/plugin.min.js',
    '/plugins/HistomicsTK/plugin_tests/client/mockVGL.js'
]);

var app;
var geojsMap;

girderTest.promise.then(function () {
    $('body').css('overflow', 'hidden');
    girder.router.enabled(false);
    girder.events.trigger('g:appload.before');
    app = new girder.plugins.HistomicsTK.App({
        el: 'body',
        parentView: null
    });
    app.bindRoutes();
    girder.events.trigger('g:appload.after');
});

$(function () {
    describe('Annotation tests', function () {
        describe('setup', function () {
            it('login', function () {
                girderTest.waitForLoad();

                runs(function () {
                    $('.g-login').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    $('#g-login').val('admin');
                    $('#g-password').val('password');
                    $('#g-login-button').click();
                });

                waitsFor(function () {
                    return $('.h-user-dropdown-link').length > 0;
                }, 'user to be logged in');
            });

            it('open image', function () {
                $('.h-open-image').click();
                girderTest.waitForDialog();

                runs(function () {
                    $('#g-root-selector').val(
                        girder.auth.getCurrentUser().id
                    ).trigger('change');
                });

                waitsFor(function () {
                    return $('#g-dialog-container .g-folder-list-link').length > 0;
                }, 'Hierarchy widget to render');

                runs(function () {
                    $('.g-folder-list-link:contains("Public")').click();
                });

                waitsFor(function () {
                    return $('.g-item-list-link').length > 0;
                }, 'item list to load');

                runs(function () {
                    $('.g-item-list-link').click();
                    $('.g-submit-button').click();
                });

                waitsFor(function () {
                    return $('.geojs-layer.active').length > 0;
                }, 'image to load');

                runs(function () {
                    geojsMap = app.bodyView.viewer;
                    window.mockVGLRenderer(true);
                });
            });
        });

        describe('Draw panel', function () {
            it('draw a point', function () {
                runs(function () {
                    $('.h-draw[data-type="point"]').click();
                });

                waitsFor(function () {
                    return $('.geojs-map.annotation-input').length > 0;
                }, 'draw mode to activate');
                runs(function () {
                    var interactor = geojsMap.interactor();
                    interactor.simulateEvent('mousedown', {
                        map: {x: 100, y: 100},
                        button: 'left'
                    });
                    interactor.simulateEvent('mouseup', {
                        map: {x: 100, y: 100},
                        button: 'left'
                    });
                });

                waitsFor(function () {
                    return $('.h-elements-container .h-element').length === 1;
                }, 'point to be created');
                runs(function () {
                    expect($('.h-elements-container .h-element .h-element-label').text()).toBe('point');
                });
            });

            it('edit a point element', function () {
                runs(function () {
                    $('.h-elements-container .h-edit-element').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    expect($('#g-dialog-container .modal-title').text()).toBe('Edit annotation');
                    $('#g-dialog-container #h-element-label').val('test');
                    $('#g-dialog-container .h-submit').click();
                });

                waitsFor(function () {
                    return $('.h-elements-container .h-element .h-element-label').text() === 'test';
                }, 'label to change');
            });

            it('draw another point', function () {
                runs(function () {
                    $('.h-draw[data-type="point"]').click();
                });

                waitsFor(function () {
                    return $('.geojs-map.annotation-input').length > 0;
                }, 'draw mode to activate');
                runs(function () {
                    var interactor = geojsMap.interactor();
                    interactor.simulateEvent('mousedown', {
                        map: {x: 200, y: 200},
                        button: 'left'
                    });
                    interactor.simulateEvent('mouseup', {
                        map: {x: 200, y: 200},
                        button: 'left'
                    });
                });

                waitsFor(function () {
                    return $('.h-elements-container .h-element').length === 2;
                }, 'rectangle to be created');
                runs(function () {
                    expect($('.h-elements-container .h-element:last .h-element-label').text()).toBe('point');
                });
            });

            it('delete the second point', function () {
                $('.h-elements-container .h-element:last .h-delete-element').click();
                expect($('.h-elements-container .h-element').length).toBe(1);
            });

            it('save the point annotation', function () {
                var annotations = null;
                runs(function () {
                    $('.h-draw-widget .h-save-annotation').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    $('#g-dialog-container #h-annotation-name').val('single point');
                    $('#g-dialog-container .h-submit').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation-name').text()).toBe('single point');
                    expect($('.h-draw-widget .h-save-widget').length).toBe(0);

                    girder.rest.restRequest({
                        path: 'annotation',
                        data: {
                            itemId: girder.plugins.HistomicsTK.router.getQuery('image')
                        }
                    }).then(function (a) { annotations = a; });
                });

                waitsFor(function () {
                    return annotations !== null;
                }, 'get annotations from server');
                runs(function () {
                    expect(annotations.length).toBe(1);
                    expect(annotations[0].annotation.name).toBe('single point');
                });
            });
        });

        describe('Annotation panel', function () {
            it('panel is rendered', function () {
                expect($('.h-annotation-selector .s-panel-title').text()).toMatch(/Annotations/);
            });

            it('toggle visibility of an annotation', function () {
                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    expect($el.length).toBe(1);
                    expect($el.find('.icon-eye-off.h-toggle-annotation').length).toBe(1);
                    $el.find('.h-toggle-annotation').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    return $el.find('.icon-eye.h-toggle-annotation').length === 1;
                }, 'annotation to toggle');
            });

            it('delete an annotation', function () {
                var annotations = null;
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("single point") .h-delete-annotation').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation:contains("single point")').length).toBe(0);
                    girder.rest.restRequest({
                        path: 'annotation',
                        data: {
                            itemId: girder.plugins.HistomicsTK.router.getQuery('image')
                        }
                    }).then(function (a) { annotations = a; });
                });

                waitsFor(function () {
                    return annotations !== null;
                }, 'get annotations from server');
                runs(function () {
                    expect(annotations.length).toBe(0);
                });
            });
        });
    });
});
