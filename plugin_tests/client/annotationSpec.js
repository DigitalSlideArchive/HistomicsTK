girderTest.importPlugin('jobs');
girderTest.importPlugin('worker');
girderTest.importPlugin('large_image');
girderTest.importPlugin('slicer_cli_web');
girderTest.importPlugin('HistomicsTK');

var app;
var geojsMap;
var imageId;

girderTest.promise.then(function () {
    $('body').css('overflow', 'hidden');
    girder.router.enabled(false);
    girder.events.trigger('g:appload.before');
    girder.plugins.HistomicsTK.panels.DrawWidget.throttleAutosave = false;
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
        waitsFor(function () {
            return $('#g-selected-model').val();
        }, 'selection to be set');

        girderTest.waitForDialog();
        runs(function () {
            $('.g-submit-button').click();
        });

        girderTest.waitForLoad();
        waitsFor(function () {
            return $('.geojs-layer.active').length > 0;
        }, 'image to load');
        runs(function () {
            expect(girder.plugins.HistomicsTK.router.getQuery('image')).toBe(imageId);
        });
    }

    /**
     * This is a test helper method to make assertions about the last autosaved
     * annotation.  The autosaved annotation is assumed to be the last annotation
     * returned by the `/api/v1/annotation` endpoint.
     *
     * @param {string} imageId The id of the currently opened image
     * @param {number} annotationName An annotation name expected to be in the result
     * @param {number|null} numberOfElements
     *      If a number, the number of elements to expect in the annotation.
     * @param {object}
     *      The annotations loaded from the server will be set in this object for
     *      further use by the caller.
     */
    function checkAutoSave(imageId, annotationName, numberOfElements, annotationInfo) {
        var annotations;
        var annotation;

        girderTest.waitForLoad();

        // If the next rest request happens too quickly after saving the
        // annotation, the database might not be synced.  Ref:
        // https://travis-ci.org/DigitalSlideArchive/HistomicsTK/builds/283691041
        waits(100);
        runs(function () {
            girder.rest.restRequest({
                url: 'annotation',
                data: {
                    itemId: imageId,
                    userId: girder.auth.getCurrentUser().id
                }
            }).then(function (a) {
                annotations = a;
                return null;
            });
        });

        waitsFor(function () {
            return annotations !== undefined;
        }, 'saved annotations to load');
        runs(function () {
            var i, foundIndex = -1;
            for (i = 0; i < annotations.length; i += 1) {
                if (annotations[i].annotation.name === annotationName) {
                    foundIndex = i;
                }
            }
            expect(foundIndex).toBeGreaterThan(-1);
            annotationInfo.annotations = annotations;
            girder.rest.restRequest({
                url: 'annotation/' + annotations[foundIndex]._id
            }).done(function (a) {
                annotationInfo.annotation = a;
                annotation = a;
            });
        });

        waitsFor(function () {
            return annotation !== undefined;
        }, 'annotation to load');
        runs(function () {
            if (numberOfElements !== null) {
                expect(annotation.annotation.elements.length).toBe(numberOfElements);
            }
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
                    $('#g-login').val('user');
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

        describe('Download view and region of interest', function () {
            it('check href attribute of \'Download View\' link', function () {
                runs(function () {
                    $('#download-view-link').bind('click', function (event) {
                        event.preventDefault();
                    });
                    $('.h-download-button-view').click();
                });

                waitsFor(function () {
                    return $('#download-view-link').attr('href') !== undefined;
                }, 'to be the url');

                runs(function () {
                    expect($('#download-view-link').attr('href')).toMatch(/\/item\/[0-9a-f]{24}\/tiles\/region\?width=[0-9-]+&height=[0-9-]+&left=[0-9-]+&top=[0-9-]+&right=[0-9-]+&bottom=[0-9-]+&contentDisposition=attachment/);
                });
            });

            it('open the download dialog', function () {
                var interactor = geojsMap.interactor();
                $('.h-download-button-area').click();

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

                girderTest.waitForDialog();
                runs(function () {
                    expect($('.modal-title').text()).toBe('Edit Area');
                });
            });

            it('test modifying form elements', function () {
                const oldSettings = [];
                const elements = [];
                elements.push($('#h-element-width'), $('#h-element-height'),
                    $('#h-nb-pixel'), $('#h-size-file'));
                oldSettings.push($('#h-element-width').val(), $('#h-element-height').val(),
                    $('#h-nb-pixel').val(), $('#h-size-file').val());
                runs(function () {
                    $('#h-element-mag').val(10).trigger('change');
                    var i = 0;
                    // Check all the setting labels change
                    for (var value in oldSettings) {
                        expect(elements[i].val()).not.toEqual(value);
                        i++;
                    }
                });
                runs(function () {
                    $('#h-download-image-format').val('TIFF').trigger('change');
                    // Check the size label change
                    expect($('#h-size-file').val()).not.toEqual(oldSettings[3]);
                });
            });

            it('ensure the download link is correct', function () {
                waitsFor(function () {
                    return $('#h-download-area-link').attr('href') !== undefined;
                }, 'to be the url');

                runs(function () {
                    expect($('#h-download-area-link').attr('href')).toMatch(/\/item\/[0-9a-f]{24}\/tiles\/region\?regionWidth=[0-9-]+&regionHeight=[0-9-]+&left=[0-9-]+&top=[0-9-]+&right=[0-9-]+&bottom=[0-9-]+&encoding=[EFGIJNPT]{3,4}&contentDisposition=attachment&magnification=[0-9-]+/);
                });
            });

            it('close the dialog', function () {
                $('#g-dialog-container').girderModal('close');
                waitsFor(function () {
                    return $('body.modal-open').length === 0;
                });
            });
        });

        describe('Draw panel', function () {
            var annotationInfo = {};

            it('create a new annotation', function () {
                $('.h-create-annotation').click();
                girderTest.waitForDialog();

                runs(function () {
                    var nameInput = $('#h-annotation-name');
                    expect(nameInput.length).toBe(1);

                    nameInput.val('drawn 1');
                    $('.h-submit').click();
                });
                girderTest.waitForLoad();

                runs(function () {
                    expect($('.h-draw-widget').hasClass('hidden')).toBe(false);
                });
            });

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
                    expect($('.h-draw[data-type="point"]').hasClass('active')).toBe(true);
                    // turn off point drawing.
                    $('.h-draw[data-type="point"]').click();
                });
                waitsFor(function () {
                    return !$('.h-draw[data-type="point"]').hasClass('active');
                }, 'point drawing to be off');

                checkAutoSave(imageId, 'drawn 1', 1, annotationInfo);
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
                }, 'point to be created');
                runs(function () {
                    expect($('.h-elements-container .h-element:last .h-element-label').text()).toBe('point');
                });
                checkAutoSave(imageId, 'drawn 1', 2, annotationInfo);
            });

            it('delete the second point', function () {
                $('.h-elements-container .h-element:last .h-delete-element').click();
                expect($('.h-elements-container .h-element').length).toBe(1);
                checkAutoSave(imageId, 'drawn 1', 1, annotationInfo);
            });

            it('draw another point', function () {
                runs(function () {
                    // The draw button must be clicked in the next event loop (not sure why).
                    window.setTimeout(function () {
                        $('.h-draw[data-type="point"]').click();
                    }, 0);
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
                    return $('.h-elements-container .h-element').length === 2;
                }, 'point to be created');
                runs(function () {
                    expect($('.h-elements-container .h-element:last .h-element-label').text()).toBe('point');
                });
                checkAutoSave(imageId, 'drawn 1', 2, annotationInfo);
            });

            it('delete the last point', function () {
                $('.h-elements-container .h-element:last .h-delete-element').click();
                expect($('.h-elements-container .h-element').length).toBe(1);

                // reset the draw state
                $('.h-draw[data-type="point"]').click();
            });
        });

        describe('Annotation styles', function () {
            it('create a new annotation', function () {
                $('.h-create-annotation').click();
                girderTest.waitForDialog();

                runs(function () {
                    var nameInput = $('#h-annotation-name');
                    expect(nameInput.length).toBe(1);

                    nameInput.val('drawn 2');
                    $('.h-submit').click();
                });
                girderTest.waitForLoad();

                runs(function () {
                    expect($('.h-draw-widget').hasClass('hidden')).toBe(false);
                });
            });

            it('open the syle group dialog', function () {
                runs(function () {
                    $('.h-configure-style-group').click();
                });
                girderTest.waitForDialog();
                waitsFor(function () {
                    return $('body.modal-open').length > 0;
                }, 'dialog to open');
                runs(function () {
                    // ensure the default style is created on load
                    expect($('.h-group-name :selected').val()).toBe('default');
                });
            });

            it('test reset to defaults as a regular user', function () {
                runs(function () {
                    $('#h-element-line-width').val('10');
                    $('#h-reset-defaults').click();
                });
                waitsFor(function () {
                    return $('#h-element-line-width').val() === '2';
                });
                runs(function () {
                    expect($('.h-group-name :selected').val()).toBe('default');
                });
            });

            it('create a new style group', function () {
                $('.h-create-new-style').click();
                $('.h-new-group-name').val('new');
                $('.h-save-new-style').click();
                expect($('.h-group-name :selected').val()).toBe('new');

                $('#h-element-line-width').val(1).trigger('change');
                $('#h-element-line-color').val('rgb(255,0,0)').trigger('change');
                $('#h-element-fill-color').val('rgb(0,0,255)').trigger('change');
                $('.h-submit').click();

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-style-group').val()).toBe('new');
                });
            });

            it('draw a point', function () {
                runs(function () {
                    // The draw button must be clicked in the next event loop (not sure why).
                    window.setTimeout(function () {
                        $('.h-draw[data-type="point"]').click();
                    }, 0);
                });
                runs(function () {
                    $('.h-draw[data-type="point"]').removeClass('active').click();
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
                    return app.bodyView.drawWidget.collection.length > 0;
                }, 'collection to update');
                runs(function () {
                    var elements = app.bodyView.drawWidget.collection;
                    expect(elements.length).toBe(1);

                    var point = elements.at(0);
                    expect(point.get('lineWidth')).toBe(1);
                    expect(point.get('lineColor')).toBe('rgb(255, 0, 0)');
                    expect(point.get('fillColor')).toBe('rgb(0, 0, 255)');
                });
            });

            it('open the syle group dialog again', function () {
                runs(function () {
                    $('.h-configure-style-group').click();
                });
                girderTest.waitForDialog();
                waitsFor(function () {
                    return $('body.modal-open').length > 0;
                }, 'dialog to open');
                runs(function () {
                    expect($('.h-group-name :selected').val()).toBe('new');
                });
            });

            it('delete a style group', function () {
                runs(function () {
                    $('.h-delete-style').click();
                    expect($('.h-group-name :selected').val()).toBe('default');
                    $('.h-submit').click();
                });
                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-style-group').val()).toBe('default');
                });
            });
        });

        describe('Annotation panel', function () {
            it('panel is rendered', function () {
                expect($('.h-annotation-selector .s-panel-title').text()).toMatch(/Annotations/);
            });

            it('ensure user cannot remove the admin annotation', function () {
                var adminAnnotation = $('.h-annotation-selector .h-annotation:contains("admin annotation")');
                expect(adminAnnotation.length).toBe(1);
                expect(adminAnnotation.find('.h-delete-annotation').length).toBe(0);
            });

            it('hide all annotations', function () {
                $('.h-hide-all-annotations').click();
                girderTest.waitForLoad();

                runs(function () {
                    expect($('.h-annotation .icon-eye-off').length).toBe(3);
                    expect($('.h-annotation .icon-eye').length).toBe(0);
                });
            });

            it('show all annotations', function () {
                $('.h-show-all-annotations').click();
                girderTest.waitForLoad();

                waitsFor(function () {
                    var $el = $('.h-annotation-selector');
                    return $el.find('.icon-spin3').length === 0;
                }, 'loading spinners to disappear');
                runs(function () {
                    expect($('.h-annotation .icon-eye-off').length).toBe(0);
                    expect($('.h-annotation .icon-eye').length).toBe(3);
                });
            });

            it('toggle visibility of an annotation', function () {
                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("drawn 1")');
                    $el.find('.h-toggle-annotation').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("drawn 1")');
                    return $el.find('.icon-eye-off.h-toggle-annotation').length === 1;
                }, 'annotation to toggle off');

                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("drawn 1")');
                    $el.find('.h-toggle-annotation').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("drawn 1")');
                    return $el.find('.icon-eye.h-toggle-annotation').length === 1;
                }, 'annotation to toggle on');
            });

            it('edit annotation metadata', function () {
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("drawn 1") .h-edit-annotation-metadata').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    expect($('#h-annotation-name').val()).toBe('drawn 1');
                    $('#h-annotation-name').val('');
                    $('#h-annotation-description').val('description');
                    $('.h-submit').click();

                    var validationEl = $('.g-validation-failed-message');
                    expect(validationEl.length).toBe(1);
                    expect(validationEl.hasClass('hidden')).toBe(false);
                    expect(validationEl.text()).toBe('Please enter a name.');

                    $('#h-annotation-name').val('edited 1');
                    $('.h-submit').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    var $el = $('.h-annotation-selector .h-annotation:contains("edited 1")');
                    expect($el.length).toBe(1);
                });
            });

            it('delete an annotation', function () {
                var annotations = null;
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("edited 1") .h-delete-annotation').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    $('.h-submit').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation:contains("edited 1")').length).toBe(0);
                    girder.rest.restRequest({
                        url: 'annotation',
                        data: {
                            itemId: imageId,
                            userId: girder.auth.getCurrentUser().id
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
                });
            });

            it('cannot edit an annotation as a non-admin', function () {
                var trigger = girder.events.trigger;
                var alertTriggered;
                girder.events.trigger = _.wrap(girder.events.trigger, function (func, event, options) {
                    if (event === 'g:alert') {
                        alertTriggered = options;
                    }
                    return func.apply(arguments);
                });

                $('.h-annotation-selector .h-annotation:contains("drawn 2") .h-toggle-annotation').click();
                $('.h-annotation-selector .h-annotation:contains("admin annotation") .h-annotation-name').click();

                girderTest.waitForLoad();
                runs(function () {
                    expect(alertTriggered).toBeDefined();
                    expect(alertTriggered.text).toBe('You do not have write access to this annotation.');
                    girder.events.trigger = trigger;
                });
            });

            it('close the open draw panel', function () {
                $('.h-annotation-selector .h-annotation:contains("drawn 2") .h-annotation-name').click();
                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-elements-container').length).toBe(0);
                });
            });

            it('open the draw panel for an editable annotation', function () {
                $('.h-annotation-selector .h-annotation:contains("drawn 2") .h-annotation-name').click();
                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-elements-container').length).toBe(1);
                    expect($('.h-annotation-selector .h-annotation:contains("drawn 2") .icon-eye').length).toBe(1);
                    expect($('.h-draw-widget .h-panel-name').text()).toBe('drawn 2');
                });
            });

            it('trigger a mouseon event on an element', function () {
                var annotation = $('.h-annotation-selector .h-annotation:contains("drawn 2")').data('id');
                var element = app.bodyView.annotations.get(annotation).elements().get($('.h-draw-widget .h-element').data('id'));
                app.bodyView.viewerWidget.trigger('g:mouseOnAnnotation', element, annotation);
                expect($('.h-annotation-selector .h-annotation:contains("drawn 2")').hasClass('h-highlight-annotation')).toBe(true);
                expect($('.h-draw-widget .h-element').hasClass('h-highlight-element')).toBe(true);
            });

            it('trigger a mouseoff event', function () {
                var annotation = $('.h-annotation-selector .h-annotation:contains("drawn 2")').data('id');
                var element = app.bodyView.annotations.get(annotation).elements().get($('.h-draw-widget .h-element').data('id'));
                app.bodyView.viewerWidget.trigger('g:mouseOffAnnotation', element, annotation);
                expect($('.h-annotation-selector .h-annotation:contains("drawn 2")').hasClass('h-highlight-annotation')).toBe(false);
                expect($('.h-draw-widget .h-element').hasClass('h-highlight-element')).toBe(false);
            });

            it('mouseover an annotation in the AnnotationSelector', function () {
                var called;
                var highlightAnnotation = app.bodyView.viewerWidget.highlightAnnotation;
                app.bodyView.viewerWidget.highlightAnnotation = function (annotation, element) {
                    called = true;
                    expect(annotation).toBe($('.h-annotation-selector .h-annotation:contains("drawn 2")').data('id'));
                    expect(element).toBeUndefined();
                };
                app.bodyView.annotationSelector.$('.h-annotation:contains("drawn 2")').trigger('mouseenter');
                expect(called).toBe(true);
                app.bodyView.viewerWidget.highlightAnnotation = highlightAnnotation;
            });

            it('mouseover an annotation in the Draw widget', function () {
                var called;
                var highlightAnnotation = app.bodyView.viewerWidget.highlightAnnotation;
                app.bodyView.viewerWidget.highlightAnnotation = function (annotation, element) {
                    called = true;
                    expect(annotation).toBe($('.h-annotation-selector .h-annotation:contains("drawn 2")').data('id'));
                    expect(element).toBe($('.h-element').data('id'));
                };
                $('.h-element').trigger('mouseenter');
                expect(called).toBe(true);
                app.bodyView.viewerWidget.highlightAnnotation = highlightAnnotation;
            });

            it('mouseout to reset the highlight state', function () {
                var called;
                var highlightAnnotation = app.bodyView.viewerWidget.highlightAnnotation;
                app.bodyView.viewerWidget.highlightAnnotation = function (annotation, element) {
                    called = true;
                    expect(annotation).toBeUndefined();
                    expect(element).toBeUndefined();
                };
                $('.h-element').trigger('mouseout');
                expect(called).toBe(true);
                app.bodyView.viewerWidget.highlightAnnotation = highlightAnnotation;
            });

            it('mouseover a hidden annotation should be a no-op', function () {
                var highlightAnnotation = app.bodyView.viewerWidget.highlightAnnotation;
                app.bodyView.viewerWidget.highlightAnnotation = function (annotation, element) {
                    throw new Error('should not be called');
                };
                $('.h-annotation-selector .h-annotation:contains("admin annotation") .icon-eye').click();

                girderTest.waitForLoad();
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("admin annotation")').trigger('mouseenter');
                    app.bodyView.viewerWidget.highlightAnnotation = highlightAnnotation;
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
                        url: 'annotation?itemId=' + imageId,
                        contentType: 'application/json',
                        processData: false,
                        data: JSON.stringify(rect),
                        method: 'POST'
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
                waitsFor(function () {
                    var $el = $('.h-annotation-selector');
                    return $el.find('.icon-spin3').length === 0;
                }, 'loading spinners to disappear');
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
                    expect($('.h-annotation-selector .h-annotation').length).toBe(3);
                });
            });
        });
    });

    describe('Open recently annotated image', function () {
        it('open the dialog', function () {
            runs(function () {
                $('.h-open-annotated-image').click();
            });
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
            runs(function () {
                var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                $el.click();
            });
            girderTest.waitForLoad();
            runs(function () {
                expect(girder.plugins.HistomicsTK.router.getQuery('image')).toBe(imageId);
            });
        });
    });

    describe('Annotation tests as admin', function () {
        describe('setup', function () {
            girderTest.logout()();
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
                girderTest.waitForLoad();
            });

            it('open the dialog', function () {
                runs(function () {
                    $('.h-open-annotated-image').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                    return $el.length === 1;
                }, 'here');
                girderTest.waitForDialog();
                runs(function () {
                    var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                    expect($el.length).toBe(1);
                    // remock VGL
                    app.bodyView.once('h:viewerWidgetCreated', function (viewerWidget) {
                        viewerWidget.once('g:beforeFirstRender', function () {
                            window.geo.util.mockVGLRenderer();
                        });
                    });
                    $el.click();
                });
                girderTest.waitForLoad();
                runs(function () {
                    expect(girder.plugins.HistomicsTK.router.getQuery('image')).toBe(imageId);
                });
            });
        });

        describe('style group tests', function () {
            it('open an annotation in the draw panel', function () {
                waitsFor(function () {
                    return $('.h-annotation-selector .h-annotation:contains("drawn 2") .h-toggle-annotation').length;
                }, 'annotations to appear');
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("admin annotation") .h-annotation-name').click();
                });
                girderTest.waitForLoad();
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("drawn 2") .h-annotation-name').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    expect($('.h-elements-container').length).toBe(1);
                    expect($('.h-annotation-selector .h-annotation:contains("drawn 2") .icon-eye').length).toBe(1);
                    expect($('.h-draw-widget .h-panel-name').text()).toBe('drawn 2');
                });
            });

            it('open the syle group dialog', function () {
                runs(function () {
                    $('.h-configure-style-group').click();
                });
                girderTest.waitForDialog();
            });

            it('set the default style groups', function () {
                runs(function () {
                    $('#h-set-defaults').click();
                });
                waitsFor(function () {
                    var resp = girder.rest.restRequest({
                        url: 'system/setting',
                        method: 'GET',
                        data: {list: JSON.stringify(['histomicstk.default_draw_styles'])},
                        async: false
                    });
                    var settings = resp.responseJSON;
                    var settingsStyles = settings && JSON.parse(settings['histomicstk.default_draw_styles']);
                    if (!settingsStyles || !settingsStyles.length) {
                        return false;
                    }
                    return settingsStyles[0].group === 'new';
                });
            });
            it('reset the style groups', function () {
                runs(function () {
                    $('.h-group-name').val('new');  // select the 'new' style
                    $('#h-element-line-width').val('10');
                    $('#h-element-label').val('newlabel');
                    $('#h-reset-defaults').click();
                });
                waitsFor(function () {
                    return $('#h-element-label').val() === '';
                }, 'label to reset');
                runs(function () {
                    expect($('#h-element-line-width').val()).toBe('2');
                });
            });
            it('cancel changes', function () {
                runs(function () {
                    $('.h-cancel').click();
                });
                girderTest.waitForLoad();
            });
        });
    });
});
