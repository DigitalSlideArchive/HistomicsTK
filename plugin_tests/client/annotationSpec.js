girderTest.importPlugin('jobs');
girderTest.importPlugin('worker');
girderTest.importPlugin('large_image');
girderTest.importPlugin('item_tasks');
girderTest.importPlugin('HistomicsTK');

var app;
var geojsMap;
var imageId;

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
    return null;
});

$(function () {
    function openImage(name) {
        runs(function () {
            app.bodyView.once('h:viewerWidgetCreated', function (viewerWidget) {
                viewerWidget.once('g:beforeFirstRender', function () {
                    window.geo.util.mockVGLRenderer();
                });
            });
            $('.h-open-image').click();
        });

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
            var $item = $('.g-item-list-link:contains("' + name + '")');
            imageId = $item.next().attr('href').match(/\/item\/([a-f0-9]+)\/download/)[1];
            expect($item.length).toBe(1);
            $item.click();
        });

        girderTest.waitForDialog();
        // Sometimes clicking submit fires the `g:saved` event, but doesn't actually
        // close the dialog.  Delaying the click seems to help.
        runs(function () {
            window.setTimeout(function () {
                $('.g-submit-button').click();
            }, 100);
        });

        girderTest.waitForLoad();
        waitsFor(function () {
            return $('.geojs-layer.active').length > 0;
        }, 'image to load');
        runs(function () {
            expect(girder.plugins.HistomicsTK.router.getQuery('image')).toBe(imageId);
        });
    }

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
                openImage('image');
                runs(function () {
                    geojsMap = app.bodyView.viewer;
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

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-elements-container .h-element .h-element-label').text()).toBe('test');
                });
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
                            itemId: imageId
                        }
                    }).then(function (a) {
                        annotations = a;
                        return null;
                    });
                });

                waitsFor(function () {
                    return annotations !== null;
                }, 'get annotations from server');
                runs(function () {
                    expect(annotations.length).toBe(1);
                    expect(annotations[0].annotation.name).toBe('single point');
                });

                waitsFor(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    return $el.find('.icon-eye.h-toggle-annotation').length === 1;
                }, 'saved annotation to draw');
            });
        });

        describe('Annotation panel', function () {
            it('panel is rendered', function () {
                expect($('.h-annotation-selector .s-panel-title').text()).toMatch(/Annotations/);
            });

            it('toggle visibility of an annotation', function () {
                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    $el.find('.h-toggle-annotation').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    return $el.find('.icon-eye-off.h-toggle-annotation').length === 1;
                }, 'annotation to toggle off');

                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    $el.find('.h-toggle-annotation').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("single point")');
                    return $el.find('.icon-eye.h-toggle-annotation').length === 1;
                }, 'annotation to toggle on');
            });

            it('delete an annotation', function () {
                var annotations = null;
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("single point") .h-delete-annotation').click();
                    expect($('.h-annotation-selector .h-annotation:contains("single point")').length).toBe(0);
                });

                girderTest.waitForLoad();
                runs(function () {
                    girder.rest.restRequest({
                        path: 'annotation',
                        data: {
                            itemId: imageId
                        }
                    }).then(function (a) {
                        annotations = a;
                        return null;
                    });
                });

                waitsFor(function () {
                    return annotations !== null;
                }, 'get annotations from server');
                runs(function () {
                    expect(annotations.length).toBe(0);
                });
            });

            it('show new annotations during job events', function () {
                var uploaded = false;

                runs(function () {
                    var rect = {
                        'name': 'rectangle',
                        'description': 'the description',
                        'elements': [
                            {
                                'center': [
                                    2000,
                                    2000,
                                    0
                                ],
                                'height': 4000,
                                'rotation': 0,
                                'type': 'rectangle',
                                'width': 4000
                            }
                        ]
                    };

                    girder.rest.restRequest({
                        path: 'annotation?itemId=' + imageId,
                        contentType: 'application/json',
                        processData: false,
                        data: JSON.stringify(rect),
                        type: 'POST'
                    }).then(function () {
                        uploaded = true;
                        return null;
                    });
                });

                waitsFor(function () {
                    return uploaded;
                }, 'annotation to be uploaded');
                runs(function () {
                    girder.utilities.eventStream.trigger('g:event.job_status', {
                        data: {status: 3}
                    });
                });

                waitsFor(function () {
                    return $('.h-annotation-selector .h-annotation:contains("rectangle")').length === 1;
                }, 'new annotation to appear');
                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("rectangle")');
                    expect($el.find('.icon-eye.h-toggle-annotation').length).toBe(1);
                });
            });

            it('hover over annotation with labels off', function () {
                girderTest.waitForLoad();
                runs(function () {
                    var interactor = geojsMap.interactor();
                    interactor.simulateEvent('mousemove', {
                        map: {x: 50, y: 50}
                    });
                    expect($('#h-annotation-popover-container').hasClass('hidden')).toBe(true);
                });
            });

            it('hover over annotation with labels on', function () {
                var done = false;
                runs(function () {
                    $('#h-toggle-labels').click();

                    // Ensure the next mouse move event happens asynchronously.
                    // Without doing this, the hover event occasionally fails to
                    // fire.
                    window.setTimeout(function () {
                        done = true;
                    }, 0);
                });

                waitsFor(function () {
                    return done;
                }, 'next event loop');

                runs(function () {
                    var interactor = geojsMap.interactor();
                    interactor.simulateEvent('mousemove', {
                        map: {x: 45, y: 45}
                    });

                    var $el = $('#h-annotation-popover-container');
                    expect($el.hasClass('hidden')).toBe(false);
                    expect($el.find('.h-annotation-name').text()).toBe('rectangle');
                    expect($el.find('.h-annotation-description').text()).toMatch(/the description/);
                });
            });

            it('open a different image', function () {
                openImage('copy');
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation').length).toBe(0);
                });
            });

            it('open the original image', function () {
                openImage('image');
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation').length).toBe(1);
                });
            });
        });
    });

    describe('Open recently annotated image', function () {
        it('open the dialog', function () {
            $('.h-open-annotated-image').click();
            girderTest.waitForDialog();
            runs(function () {
                var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                expect($el.length).toBe(1);
                expect($el.find('.media-left img').prop('src'))
                    .toMatch(/item\/[0-9a-f]*\/tiles\/thumbnail/);
                expect($el.find('.media-heading').text()).toBe('image');
            });
        });

        it('click on the image', function () {
            var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
            $el.click();
            girderTest.waitForLoad();
            runs(function () {
                expect(girder.plugins.HistomicsTK.router.getQuery('image')).toBe(imageId);
            });
        });
    });
});
