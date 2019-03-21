/* global histomicsTest */

girderTest.importPlugin('jobs');
girderTest.importPlugin('worker');
girderTest.importPlugin('large_image');
girderTest.importPlugin('slicer_cli_web');
girderTest.importPlugin('HistomicsTK');

var app;
girderTest.addScript('/plugins/HistomicsTK/plugin_tests/client/common.js');

girderTest.promise.done(function () {
    app = histomicsTest.startApp();
});

$(function () {
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
    function checkAutoSave(annotationName, numberOfElements, annotationInfo) {
        var annotations;
        var annotation;
        var imageId = histomicsTest.imageId();

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
                histomicsTest.login();
            });

            it('open image', function () {
                histomicsTest.openImage('image');
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
                var interactor = histomicsTest.geojsMap().interactor();
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
                runs(function () {
                    $('.h-create-annotation').click();
                });
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
                    var interactor = histomicsTest.geojsMap().interactor();
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

                checkAutoSave('drawn 1', 1, annotationInfo);
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
                    var interactor = histomicsTest.geojsMap().interactor();
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
                checkAutoSave('drawn 1', 2, annotationInfo);
            });

            it('delete the second point', function () {
                $('.h-elements-container .h-element:last .h-delete-element').click();
                expect($('.h-elements-container .h-element').length).toBe(1);
                checkAutoSave('drawn 1', 1, annotationInfo);
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
                    var interactor = histomicsTest.geojsMap().interactor();
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
                checkAutoSave('drawn 1', 2, annotationInfo);
            });

            it('delete the last point', function () {
                $('.h-elements-container .h-element:last .h-delete-element').click();
                expect($('.h-elements-container .h-element').length).toBe(1);

                // reset the draw state
                $('.h-draw[data-type="point"]').click();
            });

            it('check that the drawing type persists when switching annotatations', function () {
                runs(function () {
                    $('.h-annotation-selector .h-group-collapsed .h-annotation-group-name').click();
                    expect($('button.h-draw[data-type="point"]').hasClass('active')).toBe(true);
                    $('.h-create-annotation').click();
                });
                girderTest.waitForDialog();
                runs(function () {
                    $('#h-annotation-name').val('drawn b');
                    $('.h-submit').click();
                });
                girderTest.waitForLoad();
                // expect that the drawing type is the same as before
                runs(function () {
                    expect($('button.h-draw[data-type="point"]').hasClass('active')).toBe(true);
                });
                waitsFor(function () {
                    $('.h-annotation-selector .h-group-collapsed .h-annotation-group-name').click();
                    return $('.h-annotation-selector .h-annotation:contains("drawn b")').length;
                }, '"drawn b" control to be shown');

                // delete the annotation we just created.
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("drawn b") .h-delete-annotation').click();
                });
                girderTest.waitForDialog();
                runs(function () {
                    $('.h-submit').click();
                });
                girderTest.waitForLoad();
                waitsFor(function () {
                    return $('.h-annotation-selector .h-annotation:contains("drawn b")').length === 0;
                }, '"drawn b" to be deleted');
                // select the original annotation
                runs(function () {
                    expect($('.h-annotation-selector .h-annotation:contains("drawn 1") .h-annotation-name').length).toBe(1);
                    $('.h-annotation-selector .h-annotation:contains("drawn 1") .h-annotation-name').click();
                });
                waitsFor(function () {
                    return $('.h-draw-widget').not('.hidden').length;
                });
                girderTest.waitForLoad();
                // expect that the drawing type is the same as before
                runs(function () {
                    expect($('button.h-draw[data-type="point"]').hasClass('active')).toBe(true);
                });
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
                    var interactor = histomicsTest.geojsMap().interactor();

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

                // make sure all groups are expanded
                $('.h-annotation-selector .h-group-collapsed .h-annotation-group-name').click();
            });

            it('collapse an annotation group', function () {
                var $el = $('.h-annotation-selector .h-group-expanded[data-group-name="Other"]');
                expect($el.length).toBe(1);
                $el.find('.h-annotation-group-name').click();

                $el = $('.h-annotation-selector .h-annotation-group[data-group-name="Other"]');
                expect($el.hasClass('h-group-collapsed')).toBe(true);
                expect($el.hasClass('h-group-expanded')).toBe(false);
                expect($el.find('.h-annotation').length).toBe(0);
            });

            it('expand an annotation group', function () {
                var $el = $('.h-annotation-selector .h-group-collapsed[data-group-name="Other"]');
                expect($el.length).toBe(1);
                $el.find('.h-annotation-group-name').click();

                $el = $('.h-annotation-selector .h-annotation-group[data-group-name="Other"]');
                expect($el.hasClass('h-group-collapsed')).toBe(false);
                expect($el.hasClass('h-group-expanded')).toBe(true);
                expect($el.find('.h-annotation').length).toBeGreaterThan(0);
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

            it('set the global annotation opacity', function () {
                var opacity;
                var setGlobalAnnotationOpacityFunc = app.bodyView.viewerWidget.setGlobalAnnotationOpacity;
                app.bodyView.viewerWidget.setGlobalAnnotationOpacity = function (_opacity) {
                    opacity = _opacity;
                    return setGlobalAnnotationOpacityFunc.apply(this, arguments);
                };

                $('#h-annotation-opacity').val(0.5).trigger('input');
                expect(opacity).toBe('0.5');

                app.bodyView.viewerWidget.setGlobalAnnotationOpacity = setGlobalAnnotationOpacityFunc;
            });

            it('set the global annotation fill opacity', function () {
                var opacity;
                var setGlobalAnnotationFillOpacityFunc = app.bodyView.viewerWidget.setGlobalAnnotationFillOpacity;
                app.bodyView.viewerWidget.setGlobalAnnotationFillOpacity = function (_opacity) {
                    opacity = _opacity;
                    return setGlobalAnnotationFillOpacityFunc.apply(this, arguments);
                };

                $('#h-annotation-fill-opacity').val(0.5).trigger('input');
                expect(opacity).toBe('0.5');

                app.bodyView.viewerWidget.setGlobalAnnotationFillOpacity = setGlobalAnnotationFillOpacityFunc;
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

            it('edit annotation style', function () {
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("drawn 2") .h-edit-annotation-metadata').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    expect($('#h-annotation-name').val()).toBe('drawn 2');
                    expect($('#h-annotation-line-color').length).toBe(1);
                    expect($('#h-annotation-fill-color').length).toBe(1);
                    $('#h-annotation-line-color').val('black');
                    $('#h-annotation-fill-color').val('white');
                    $('.h-submit').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    var annotation = app.bodyView.annotations.filter(function (annotation) {
                        return annotation.get('annotation').name === 'drawn 2';
                    })[0];
                    expect(annotation.get('annotation').elements[0].lineColor).toBe('rgb(0, 0, 0)');
                    expect(annotation.get('annotation').elements[0].fillColor).toBe('rgb(255, 255, 255)');
                });
            });

            it('set annotation permissions to WRITE', function () {
                runs(function () {
                    $('.h-annotation-selector .h-annotation:contains("edited 1") .h-edit-annotation-metadata').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    expect($('#g-dialog-container .h-access').length).toBe(1);
                    $('#g-dialog-container .h-access').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    var $el = $('#g-dialog-container');
                    expect($el.find('.modal-title').text()).toBe('Access control');

                    // set edit-only access
                    expect($el.find('.g-user-access-entry').length).toBe(1);
                    $el.find('.g-user-access-entry .g-access-col-right select').val(1);
                    $el.find('.g-save-access-list').click();
                });

                girderTest.waitForLoad();
                runs(function () {
                    // the user should still have the edit button
                    expect($('.h-annotation-selector .h-annotation:contains("edited 1") .h-edit-annotation-metadata').length).toBe(1);
                    $('.h-annotation-selector .h-annotation:contains("edited 1") .h-edit-annotation-metadata').click();
                });

                girderTest.waitForDialog();
                runs(function () {
                    // admin access should be removed
                    expect($('#g-dialog-container .h-access').length).toBe(0);
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
                            itemId: histomicsTest.imageId(),
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

            it('trigger a mouseon event on an element with interactivity off', function () {
                var annotation = $('.h-annotation-selector .h-annotation:contains("drawn 2")').data('id');
                var element = app.bodyView.annotations.get(annotation).elements().get($('.h-draw-widget .h-element').data('id'));
                app.bodyView.viewerWidget.trigger('g:mouseOnAnnotation', element, annotation);
                expect($('.h-annotation-selector .h-annotation:contains("drawn 2")').hasClass('h-highlight-annotation')).toBe(false);
                expect($('.h-draw-widget .h-element').hasClass('h-highlight-element')).toBe(false);
            });

            it('trigger a mouseon event and then turn interactivity off', function () {
                $('#h-toggle-interactive').click(); // interactive on
                var annotation = $('.h-annotation-selector .h-annotation:contains("drawn 2")').data('id');
                var element = app.bodyView.annotations.get(annotation).elements().get($('.h-draw-widget .h-element').data('id'));
                app.bodyView.viewerWidget.trigger('g:mouseOnAnnotation', element, annotation);
                expect($('.h-annotation-selector .h-annotation:contains("drawn 2")').hasClass('h-highlight-annotation')).toBe(true);
                expect($('.h-draw-widget .h-element').hasClass('h-highlight-element')).toBe(true);

                $('#h-toggle-interactive').click(); // interactive off
                expect($('.h-annotation-selector .h-annotation:contains("drawn 2")').hasClass('h-highlight-annotation')).toBe(false);
                expect($('.h-draw-widget .h-element').hasClass('h-highlight-element')).toBe(false);
            });

            it('turn on interactivity and trigger a mouseon event', function () {
                $('#h-toggle-interactive').click(); // interactive on
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
                        url: 'annotation?itemId=' + histomicsTest.imageId(),
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
                    var interactor = histomicsTest.geojsMap().interactor();
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
                    var interactor = histomicsTest.geojsMap().interactor();
                    interactor.simulateEvent('mousemove', {
                        map: {x: 45, y: 45}
                    });

                    var $el = $('#h-annotation-popover-container');
                    expect($el.hasClass('hidden')).toBe(false);
                    expect($el.find('.h-annotation-name').text()).toBe('rectangle');
                    expect($el.find('.h-annotation-description').text()).toMatch(/the description/);
                });
            });

            it('open and close the context menu', function () {
                var interactor = histomicsTest.geojsMap().interactor();
                interactor.simulateEvent('mousedown', {
                    map: {x: 50, y: 50},
                    button: 'right'
                });
                interactor.simulateEvent('mouseup', {
                    map: {x: 50, y: 50},
                    button: 'right'
                });

                waitsFor(function () {
                    return $('#h-annotation-context-menu').hasClass('hidden') === false;
                }, 'context menu to be shown');
                runs(function () {
                    $(document).trigger('mousedown');
                    $(document).trigger('mouseup');
                    expect($('#h-annotation-context-menu').hasClass('hidden')).toBe(true);
                });
            });

            it('delete an element from the context menu', function () {
                var interactor = histomicsTest.geojsMap().interactor();
                interactor.simulateEvent('mousedown', {
                    map: {x: 50, y: 50},
                    button: 'right'
                });
                interactor.simulateEvent('mouseup', {
                    map: {x: 50, y: 50},
                    button: 'right'
                });

                waitsFor(function () {
                    return $('#h-annotation-context-menu').hasClass('hidden') === false;
                }, 'context menu to be shown');

                // wait for the next animation frame so that the highlighting is finished
                waits(30);
                runs(function () {
                    $('#h-annotation-context-menu .h-remove-elements').click();
                    expect($('#h-annotation-context-menu').hasClass('hidden')).toBe(true);
                });
            });

            it('open a different image', function () {
                histomicsTest.waitsForPromise(
                    histomicsTest.openImage('copy').done(function () {
                        expect($('.h-annotation-selector .h-annotation').length).toBe(0);
                    }), 'Annotation selector to appear'
                );
            });

            it('open the original image', function () {
                histomicsTest.waitsForPromise(
                    histomicsTest.openImage('image').done(function () {
                        expect($('.h-annotation-selector .h-annotation').length).toBe(3);
                    }), 'Annotation selector to appear'
                );
            });
        });
    });

    describe('Open recently annotated image', function () {
        var restPromise = null;
        var girderRestRequest = null;

        beforeEach(function () {
            restPromise = $.Deferred();
            girderRestRequest = girder.rest.restRequest;
            // Wrap girder's restRequest method to notify the testing code below
            // that the image list endpoint has returned.
            girder.rest.restRequest = function (opts) {
                var promise = girderRestRequest.apply(this, arguments);
                if (opts.url === 'annotation/images') {
                    promise.done(function () {
                        restPromise.resolve(opts);
                    });
                }
                return promise;
            };
        });

        afterEach(function () {
            girder.rest.restRequest = girderRestRequest;
            restPromise = null;
        });

        it('open the dialog', function () {
            runs(function () {
                $('.h-open-annotated-image').click();
            });

            waitsFor(function () {
                var imageId = histomicsTest.imageId();
                var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                return $el.length === 1;
            });
            runs(function () {
                var imageId = histomicsTest.imageId();
                var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                expect($el.find('.media-left img').prop('src'))
                    .toMatch(/item\/[0-9a-f]*\/tiles\/thumbnail/);
                expect($el.find('.media-heading').text()).toBe('image');
            });
        });

        it('assert user list exists', function () {
            var options = $('#h-annotation-creator option');
            expect(options.length).toBe(3);
            expect(options[0].text).toBe('Any user');
            expect(options[1].text).toBe('admin');
            expect(options[2].text).toBe('user');
        });

        it('filter by creator', function () {
            var select = $('#h-annotation-creator');
            var userid = select.find('option:nth(1)').val();
            select.val(userid).trigger('change');

            histomicsTest.waitsForPromise(
                restPromise.done(function (opts) {
                    expect(opts.data.creatorId).toBe(userid);
                }),
                'Creator filter to update'
            );

            waitsFor(function () {
                return $('.h-annotated-image').length === 1;
            }, 'Dialog to rerender');
        });

        it('filter by name', function () {
            $('#h-image-name').val('invalid name').trigger('keyup');

            histomicsTest.waitsForPromise(
                restPromise.done(function (opts) {
                    expect(opts.data.imageName).toBe('invalid name');
                }),
                'Name filter to update'
            );

            waitsFor(function () {
                return $('.h-annotated-image').length === 0;
            }, 'Dialog to rerender');
        });

        it('reset filter', function () {
            $('#h-image-name').val('').trigger('keyup');
            histomicsTest.waitsForPromise(restPromise, 'Name filter to reset');

            waitsFor(function () {
                return $('.h-annotated-image').length === 1;
            }, 'Dialog to rerender');
        });

        it('click on the image', function () {
            runs(function () {
                var imageId = histomicsTest.imageId();
                var $el = $('.h-annotated-image[data-id="' + imageId + '"]');
                $el.click();
            });
            girderTest.waitForLoad();
            runs(function () {
                var imageId = histomicsTest.imageId();
                expect(girder.plugins.HistomicsTK.router.getQuery('image')).toBe(imageId);
            });
        });
    });

    describe('Annotation tests as admin', function () {
        describe('setup', function () {
            girderTest.logout()();
            it('login', function () {
                histomicsTest.login('admin', 'password');
            });

            it('open the dialog', function () {
                runs(function () {
                    $('.h-open-annotated-image').click();
                });
                waitsFor(function () {
                    var $el = $('.h-annotated-image[data-id="' + histomicsTest.imageId() + '"]');
                    return $el.length === 1;
                }, 'here');
                girderTest.waitForDialog();
                runs(function () {
                    var $el = $('.h-annotated-image[data-id="' + histomicsTest.imageId() + '"]');
                    expect($el.length).toBe(1);
                    // remock Webgl
                    app.bodyView.once('h:viewerWidgetCreated', function (viewerWidget) {
                        viewerWidget.once('g:beforeFirstRender', function () {
                            window.geo.util.mockWebglRenderer();
                        });
                    });
                    $el.click();
                });
                girderTest.waitForLoad();
                runs(function () {
                    var imageId = histomicsTest.imageId();
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
                    $('.h-group-name').val('new'); // select the 'new' style
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
