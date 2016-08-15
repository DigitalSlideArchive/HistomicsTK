if (!String.prototype.startsWith) {
    String.prototype.startsWith = function(searchString, position){
      position = position || 0;
      return this.substr(position, searchString.length) === searchString;
  };
}

_.each([
    '/plugins/HistomicsTK/node_modules/sinon/pkg/sinon.js',
    '/plugins/HistomicsTK/web_client/js/ext/backbone.localStorage.js',
    '/plugins/HistomicsTK/web_client/js/ext/bootstrap-colorpicker.js',
    '/plugins/HistomicsTK/web_client/js/ext/bootstrap-slider.js',
    '/plugins/HistomicsTK/web_client/js/ext/tinycolor.js',
    '/clients/web/static/built/plugins/large_image/geo.min.js'
], function (src) {
    $('<script/>', {src: src}).appendTo('head');
});

window.histomicstk = {};
girderTest.addCoveredScripts([
    '/clients/web/static/built/plugins/large_image/plugin.min.js',
    '/plugins/HistomicsTK/web_client/js/0init.js',
    '/plugins/HistomicsTK/web_client/js/app.js',
    '/plugins/HistomicsTK/web_client/js/models/widget.js',
    '/plugins/HistomicsTK/web_client/js/schema/parser.js',
    '/plugins/HistomicsTK/web_client/js/views/0panel.js',
    '/plugins/HistomicsTK/web_client/js/views/body.js',
    '/plugins/HistomicsTK/web_client/js/views/browserPanel.js',
    '/plugins/HistomicsTK/web_client/js/views/controlsPanel.js',
    '/plugins/HistomicsTK/web_client/js/views/controlWidget.js',
    '/plugins/HistomicsTK/web_client/js/views/itemSelectorWidget.js',
    '/plugins/HistomicsTK/web_client/js/views/header.js',
    '/plugins/HistomicsTK/web_client/js/views/jobsPanel.js',
    '/plugins/HistomicsTK/web_client/js/views/panelGroup.js',
    '/plugins/HistomicsTK/web_client/js/views/visualization.js',
    '/plugins/HistomicsTK/web_client/js/views/annotationSelectorWidget.js',
    '/clients/web/static/built/plugins/HistomicsTK/templates.js'
]);


girderTest.importStylesheet(
    '/clients/web/static/built/plugins/HistomicsTK/plugin.min.css'
);

describe('visualization', function () {
    var $el, parentView = _.extend({
        registerChildView: function () {},
        unregisterChildView: function () {}
    }, new Backbone.View());

    girder.eventStream = Object.create(Backbone.Events);

    beforeEach(function () {
        $el = $('<div/>').css({
            width: '500px',
            height: '500px'
        }).attr('id', 'h-vis-container').appendTo('body');
    });
    afterEach(function () {
        $el.remove();
    });
    it('render the view', function () {
        new histomicstk.views.Visualization({
            parentView: parentView,
            el: $el
        }).render();

        expect($el.find('.h-visualization-body').length).toBe(1);
        expect($el.find('.h-panel-group').length).toBe(1);
    });
    it('render test tiles', function () {
        var view = new histomicstk.views.Visualization({
            parentView: parentView,
            el: $el
        }).render();

        var layer, failed = false;
        view.addItem({id: 'test'})
            .then(function (_layer) {
                layer = _layer;
            })
            .fail(function (err) {
                expect('Rendering tile set failed with: ' + err).toBe(null);
                failed = true;
            });
        waitsFor(function () {
            return (layer && Object.keys(layer.activeTiles).length === 5) || failed;
        });

        runs(function () {
            var destroyed = false;
            var spy = sinon.spy(view._map, 'exit');
            view._controlModel.on('destroy', function () {
                destroyed = true;
            });
            view.destroy();

            expect(destroyed).toBe(true);
            expect(spy.called).toBe(true);
        });
    });

    it('addTileLayer without url', function () {
        var view = new histomicstk.views.Visualization({
            parentView: parentView,
            el: $el
        }).render();
        expect(function () {
            view.addTileLayer({});
        }).toThrow();
    });

    describe('bounds url parameters', function () {
        var view;
        beforeEach(function () {
            var loaded = false;
            sinon.stub(histomicstk.router, 'setQuery');
            sinon.stub(histomicstk.router, 'getQuery')
                .returns('0,100000,0,-100000');
            view = new histomicstk.views.Visualization({
                parentView: parentView,
                el: $el
            }).render();

            view.addItem({id: 'test'}).then(function (layer) {
                layer.onIdle(function () { window.setTimeout(function () { loaded = true; }, 500)});
            });
            waitsFor(function () { return loaded; });
        });
        afterEach(function () {
            view.destroy();
            histomicstk.router.setQuery.restore();
            histomicstk.router.getQuery.restore();
        });
        it('initial bounds', function () {
            var bounds = view._map.bounds();
            expect(bounds.left).toBeCloseTo(0, 6);
            expect(bounds.right).toBeCloseTo(100000, 6);
            expect(bounds.top).toBeCloseTo(-100000, 6);
            expect(bounds.bottom).toBeCloseTo(0, 6);
        });
        it('after pan', function () {
            view._map.pan({x: -10, y: 0});
            sinon.assert.calledOnce(histomicstk.router.setQuery);
            expect(histomicstk.router.setQuery.getCall(0).args[0]).toBe('bounds');
            var params = histomicstk.router.setQuery.getCall(0).args[1].split(',');
            var bounds = view._map.bounds(undefined, null);

            expect(bounds.left).toBeCloseTo(+params[0], 6);
            expect(bounds.right).toBeCloseTo(+params[1], 6);
            expect(bounds.top).toBeCloseTo(+params[2], 6);
            expect(bounds.bottom).toBeCloseTo(+params[3], 6);
        });
    });

    describe('mocked rest requests', function () {
        var server, rest, img, _img;

        beforeEach(function () {
            // manually stub becuase of a phantomjs bug: https://github.com/sinonjs/sinon/issues/329
            _img = window.Image;
            img = sinon.spy(function () {
                return sinon.createStubInstance(Image);
            });
            window.Image = img;
            rest = sinon.stub(girder, 'restRequest');
            server = sinon.fakeServer.create({
                autoRespond: false
            });
            geo.gl.vglRenderer.supported = function () {return false;};
        });

        afterEach(function () {
            server.restore();
            rest.restore();
            window.Image = _img;
        });

         it('render test image', function () {

            rest.onCall(0).returns($.when([{
                mimeType: 'image/jpeg',
                '_id': 'some image'
            }]));

            var view = new histomicstk.views.Visualization({
                parentView: parentView,
                el: $el
            }).render();

            var model = new Backbone.Model();

            view.addItem(model)
                .then(function (quad) {
                    var data = quad.data()[0];
                    expect(data.ll).toEqual({x: 0, y: 256});
                    expect(data.ur).toEqual({x: 256, y: 0});
                    expect(data.image).toMatch(/\/file\/some image\/download$/);
                })
                .fail(function (err) {
                    expect('Rendering tile set failed with: ' + err).toBe(null);
                });

            expect(img.firstCall.returnValue.src).toMatch(/\/file\/some image\/download$/);
            img.firstCall.returnValue.width = 256;
            img.firstCall.returnValue.height = 256;
            img.firstCall.returnValue.onload();

            expect(img.getCall(1).returnValue.src).toMatch(/\/file\/some image\/download$/);
        });

         it('render image from control widget', function () {
            rest.returns($.when({}));
            rest.onCall(1).returns($.when([{
                mimeType: 'image/jpeg',
                '_id': 'some image'
            }]));

            var view = new histomicstk.views.Visualization({
                parentView: parentView,
                el: $el
            }).render();

            view._controlModel.set({
                value: new Backbone.Model({
                    id: 'abcdef',
                    _modelType: 'item',
                    name: 'image.png'
                })
            });

            expect(img.firstCall.returnValue.src).toMatch(/\/file\/some image\/download$/);
            img.firstCall.returnValue.width = 256;
            img.firstCall.returnValue.height = 256;
            img.firstCall.returnValue.onload();

            expect(img.getCall(1).returnValue.src).toMatch(/\/file\/some image\/download$/);
        });

        it('invalid image from control widget', function () {
            var spy = sinon.spy();
            rest.returns($.when({}));
            rest.onCall(1).returns($.when([{
                mimeType: 'image/jpeg',
                '_id': 'some image'
            }]));

            sinon.stub(histomicstk.views.Visualization.prototype, 'addItem')
                .returns(new $.Deferred().reject().promise());

            var view = new histomicstk.views.Visualization({
                parentView: parentView,
                el: $el
            })

            girder.events.on('g:alert', spy);
            view.render()._controlModel.set({
                value: new Backbone.Model({
                    id: 'abcdef',
                    _modelType: 'item',
                    name: 'image.png'
                })
            });

            histomicstk.views.Visualization.prototype.addItem.restore();
            // TODO: Something about the test environment is resetting the error class
            // It does work correctly in the real environment.
            // expect(view.$('[data-type="file"]').hasClass('has-error')).toBe(true);
            expect(img.callCount).toBe(0);
            expect(spy.callCount).toBeGreaterThan(0);
            sinon.assert.calledWith(spy, sinon.match({
                text: 'Could not render item as an image',
                type: 'danger'
            }));
        });

        it('item with no files', function () {
            rest.onCall(0).returns($.when([]));
            rest.onCall(1).returns($.when([]));
            rest.onCall(2).returns($.when([]));
            
            var view = new histomicstk.views.Visualization({
                parentView: parentView,
                el: $el
            }).render();

            var failed = false;
            view.addItem(new girder.models.ItemModel())
                .then(function () {
                    expect('Rendering an item with no files should fail').toBe(null);
                })
                .fail(function (err) {
                    failed = true;
                    expect(err).toBe('No renderable image file found');
                });

            expect(failed).toBe(true);
        });

        it('invalid image file', function () {
            rest.onCall(0).returns($.when([]));
            rest.onCall(1).returns($.when([{
                name: 'image.jpg',
                mimeType: 'image/jpeg',
                '_id': 'some image'
            }]));

            var view = new histomicstk.views.Visualization({
                parentView: parentView,
                el: $el
            }).render();

            view.addItem(new girder.models.ItemModel({_id: 'abcdef', name: 'image.jpg'}))
                .then(function () {
                    expect('Rendering an invalid image should fail').toBe(null);
                })
                .fail(function (err) {
                    expect(err).toBe('Could not load image');
                });

            img.firstCall.returnValue.onerror();
        });
    });
});
