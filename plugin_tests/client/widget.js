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

describe('widget model', function () {
    // test different widget types
    it('range', function () {
        var w = new histomicstk.models.Widget({
            type: 'range',
            title: 'Range widget',
            min: -10,
            max: 10,
            step: 0.5
        });
        expect(w.isNumeric()).toBe(true);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        w.set('value', '0.5');
        expect(w.value()).toBe(0.5);
        expect(w.isValid()).toBe(true);

        w.set('value', 'a number');
        expect(w.isValid()).toBe(false);

        w.set('value', -11);
        expect(w.isValid()).toBe(false);

        w.set('value', 0.75);
        expect(w.isValid()).toBe(false);

        w.set('value', 0);
        expect(w.isValid()).toBe(true);
    });
    it('basic number', function () {
        var w = new histomicstk.models.Widget({
            type: 'number',
            title: 'Number widget'
        });
        expect(w.isNumeric()).toBe(true);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        w.set('value', '0.5');
        expect(w.value()).toBe(0.5);
        expect(w.isValid()).toBe(true);

        w.set('value', 'a number');
        expect(w.isValid()).toBe(false);
    });
    it('integer number', function () {
        var w = new histomicstk.models.Widget({
            type: 'number',
            title: 'Number widget',
            step: 1
        });
        w.set('value', '0.5');
        expect(w.value()).toBe(0.5);
        expect(w.isValid()).toBe(false);

        w.set('value', '-11');
        expect(w.isValid()).toBe(true);
    });
    it('float number', function () {
        var w = new histomicstk.models.Widget({
            type: 'number',
            title: 'Number widget'
        });
        w.set('value', '1e-10');
        expect(w.value()).toBe(1e-10);
        expect(w.isValid()).toBe(true);
    });
    it('boolean', function () {
        var w = new histomicstk.models.Widget({
            type: 'boolean',
            title: 'Boolean widget'
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(true);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        expect(w.value()).toBe(false);
        expect(w.isValid()).toBe(true);

        w.set('value', {});
        expect(w.value()).toBe(true);
        expect(w.isValid()).toBe(true);
    });
    it('string', function () {
        var w = new histomicstk.models.Widget({
            type: 'string',
            title: 'String widget',
            value: 'Default value'
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        expect(w.value()).toBe('Default value');
        expect(w.isValid()).toBe(true);

        w.set('value', 1);
        expect(w.value()).toBe('1');
        expect(w.isValid()).toBe(true);
    });
    it('color', function () {
        var w = new histomicstk.models.Widget({
            type: 'color',
            title: 'Color widget'
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(true);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        w.set('value', '#ffffff');
        expect(w.value()).toBe('#ffffff');
        expect(w.isValid()).toBe(true);

        w.set('value', 'red');
        expect(w.value()).toBe('#ff0000');
        expect(w.isValid()).toBe(true);

        w.set('value', 'rgb(0, 255, 0)');
        expect(w.value()).toBe('#00ff00');
        expect(w.isValid()).toBe(true);

        w.set('value', [255, 255, 0]);
        expect(w.value()).toBe('#ffff00');
        expect(w.isValid()).toBe(true);
    });
    it('string-vector', function () {
        var w = new histomicstk.models.Widget({
            type: 'string-vector',
            title: 'String vector widget'
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(true);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        w.set('value', 'a,b,c');
        expect(w.value()).toEqual(['a', 'b', 'c']);
        expect(w.isValid()).toBe(true);

        w.set('value', ['a', 1, '2']);
        expect(w.value()).toEqual(['a', '1', '2']);
        expect(w.isValid()).toBe(true);
    });
    it('number-vector', function () {
        var w = new histomicstk.models.Widget({
            type: 'number-vector',
            title: 'Number vector widget'
        });
        expect(w.isNumeric()).toBe(true);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(true);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        w.set('value', 'a,b,c');
        expect(w.isValid()).toBe(false);

        w.set('value', ['a', 1, '2']);
        expect(w.isValid()).toBe(false);

        w.set('value', '1,2,3');
        expect(w.value()).toEqual([1, 2, 3]);
        expect(w.isValid()).toBe(true);

        w.set('value', ['0', 1, '2']);
        expect(w.value()).toEqual([0, 1, 2]);
        expect(w.isValid()).toBe(true);
    });
    it('string-enumeration', function () {
        var w = new histomicstk.models.Widget({
            type: 'string-enumeration',
            title: 'String enumeration widget',
            values: [
                'value 1',
                'value 2',
                'value 3'
            ]
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(true);
        expect(w.isFile()).toBe(false);

        w.set('value', 'value 1');
        expect(w.isValid()).toBe(true);

        w.set('value', 'value 4');
        expect(w.isValid()).toBe(false);

        w.set('value', 'value 3');
        expect(w.isValid()).toBe(true);
    });
    it('number-enumeration', function () {
        var w = new histomicstk.models.Widget({
            type: 'number-enumeration',
            title: 'Number enumeration widget',
            values: [
                11,
                12,
                '13'
            ]
        });
        expect(w.isNumeric()).toBe(true);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(true);
        expect(w.isFile()).toBe(false);

        w.set('value', '11');
        expect(w.isValid()).toBe(true);

        w.set('value', 0);
        expect(w.isValid()).toBe(false);

        w.set('value', 13);
        expect(w.isValid()).toBe(true);
    });
    it('file', function () {
        var w = new histomicstk.models.Widget({
            type: 'file',
            title: 'File widget'
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(true);
    });
    it('invalid', function () {
        var w = new histomicstk.models.Widget({
            type: 'invalid type',
            title: 'Invalid widget'
        });
        expect(w.isNumeric()).toBe(false);
        expect(w.isBoolean()).toBe(false);
        expect(w.isVector()).toBe(false);
        expect(w.isColor()).toBe(false);
        expect(w.isEnumeration()).toBe(false);
        expect(w.isFile()).toBe(false);

        expect(w.isValid()).toBe(false);
    });
});

describe('widget collection', function () {
    it('values', function () {
        var c = new histomicstk.collections.Widget([
            {type: 'range', id: 'range', value: 0},
            {type: 'number', id: 'number', value: '1'},
            {type: 'boolean', id: 'boolean', value: 'yes'},
            {type: 'string', id: 'string', value: 0},
            {type: 'color', id: 'color', value: 'red'},
            {type: 'string-vector', id: 'string-vector', value: 'a,b,c'},
            {type: 'number-vector', id: 'number-vector', value: '1,2,3'},
            {type: 'string-enumeration', id: 'string-enumeration', values: ['a'], value: 'a'},
            {type: 'number-enumeration', id: 'number-enumeration', values: [1], value: '1'},
            {type: 'file', id: 'file', value: 'a'}
        ]);

        expect(c.values()).toEqual({
            range: 0,
            number: 1,
            boolean: true,
            string: '0',
            color: '#ff0000',
            'string-vector': ['a', 'b', 'c'],
            'number-vector': [1, 2, 3],
            'string-enumeration': 'a',
            'number-enumeration': 1,
            file: 'a'
        });
    });
});

describe('control widget view', function () {
    var $el, parentView = {
        registerChildView: function () {}
    };

    function checkWidgetCommon(widget) {
        var model = widget.model;
        expect(widget.$('label[for="' + model.id + '"]').text())
            .toBe(model.get('title'));
        if (widget.model.isVector()) {
            expect(widget.$('#' + model.id + ' input').length).toBe(3);
            expect(widget.$('input#' + model.id + '-0').length).toBe(1);
            expect(widget.$('input#' + model.id + '-1').length).toBe(1);
            expect(widget.$('input#' + model.id + '-2').length).toBe(1);
        } else if (widget.model.isEnumeration()) {
            expect(widget.$('select#' + model.id).length).toBe(1);
        } else {
            expect(widget.$('input#' + model.id).length).toBe(1);
        }
    }

    beforeEach(function () {
        $el = $('<div/>').appendTo('body');
    });
    afterEach(function () {
        $el.remove();
    });

    it('range', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'range',
                title: 'Title',
                id: 'range-widget',
                value: 2,
                min: 0,
                max: 10,
                step: 2
            })
        });

        w.render();

        checkWidgetCommon(w);
        expect(w.$('input').val()).toBe('2');
        w.$('input').val('4').trigger('change');
        expect(w.model.value()).toBe(4);
    });

    it('number', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'number',
                title: 'Title',
                id: 'number-widget',
                value: 2,
                min: 0,
                max: 10,
                step: 2
            })
        });

        w.render();

        checkWidgetCommon(w);
        expect(w.$('input').val()).toBe('2');
        w.$('input').val('4').trigger('change');
        expect(w.model.value()).toBe(4);

        w.$('input').val('not a number').trigger('change');
        expect(w.$('.form-group').hasClass('has-error')).toBe(true);

        w.$('input').val('4').trigger('change');
        expect(w.$('.form-group').hasClass('has-error')).toBe(false);
    });

    it('boolean', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'boolean',
                title: 'Title',
                id: 'boolean-widget'
            })
        });

        w.render();

        checkWidgetCommon(w);
        expect(w.$('input').prop('checked')).toBe(false);

        w.$('input').click();
        expect(w.model.value()).toBe(true);
    });

    it('string', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'string',
                title: 'Title',
                id: 'string-widget',
                value: 'default'
            })
        });

        w.render();

        checkWidgetCommon(w);
        expect(w.$('input').val()).toBe('default');

        w.$('input').val('new value').trigger('change');
        expect(w.model.value()).toBe('new value');
    });

    it('color', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'color',
                title: 'Title',
                id: 'color-widget',
                value: 'red'
            })
        });

        w.render();

        checkWidgetCommon(w);
        expect(w.model.value()).toBe('#ff0000');

        w.$('.input-group-addon').click();
        expect($('.colorpicker-visible').length).toBe(1);

        w.$('input').val('#ffffff').trigger('change');
        expect(w.model.value()).toBe('#ffffff');
    });

    it('string-vector', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'string-vector',
                title: 'Title',
                id: 'string-vector-widget',
                value: 'one,two,three'
            })
        });

        w.render();
        checkWidgetCommon(w);
        expect(w.$('input[data-vector-component="0"]').val()).toBe('one');
        expect(w.$('input[data-vector-component="1"]').val()).toBe('two');
        expect(w.$('input[data-vector-component="2"]').val()).toBe('three');

        w.$('input[data-vector-component="0"]').val('1').trigger('change');
        expect(w.model.value()).toEqual(['1', 'two', 'three']);
        w.$('input[data-vector-component="1"]').val('2').trigger('change');
        expect(w.model.value()).toEqual(['1', '2', 'three']);
        w.$('input[data-vector-component="2"]').val('3').trigger('change');
        expect(w.model.value()).toEqual(['1', '2', '3']);
    });

    it('number-vector', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'number-vector',
                title: 'Title',
                id: 'number-vector-widget',
                value: '1,2,3'
            })
        });

        w.render();
        checkWidgetCommon(w);
        expect(w.$('input[data-vector-component="0"]').val()).toBe('1');
        expect(w.$('input[data-vector-component="1"]').val()).toBe('2');
        expect(w.$('input[data-vector-component="2"]').val()).toBe('3');

        w.$('input[data-vector-component="0"]').val('10').trigger('change');
        expect(w.model.value()).toEqual([10, 2, 3]);
        w.$('input[data-vector-component="1"]').val('20').trigger('change');
        expect(w.model.value()).toEqual([10, 20, 3]);
        w.$('input[data-vector-component="2"]').val('30').trigger('change');
        expect(w.model.value()).toEqual([10, 20, 30]);

        w.$('input[data-vector-component="2"]').val('not a number').trigger('change');
        expect(w.$('.form-group').hasClass('has-error')).toBe(true);

        w.$('input[data-vector-component="2"]').val('1').trigger('change');
        expect(w.$('.form-group').hasClass('has-error')).toBe(false);
    });

    it('string-enumeration', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'string-enumeration',
                title: 'Title',
                id: 'string-enumeration-widget',
                value: 'value 2',
                values: [
                    'value 1',
                    'value 2',
                    'value 3'
                ]
            })
        });

        w.render();
        checkWidgetCommon(w);
        expect(w.$('select').val()).toBe('value 2');

        w.$('select').val('value 3').trigger('change');
        expect(w.model.value()).toBe('value 3');
    });

    it('number-enumeration', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'number-enumeration',
                title: 'Title',
                id: 'number-enumeration-widget',
                value: 200,
                values: [
                    100,
                    200,
                    300
                ]
            })
        });

        w.render();
        checkWidgetCommon(w);
        expect(w.$('select').val()).toBe('200');

        w.$('select').val('300').trigger('change');
        expect(w.model.value()).toBe(300);
    });

    it('file', function () {
        var arg, item = new Backbone.Model({id: 'model id'});
        var hwidget = girder.views.HierarchyWidget;
        girder.views.HierarchyWidget = Backbone.View.extend({
            initialize: function (_arg) {
                arg = _arg;
            }
        });
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'file',
                title: 'Title',
                id: 'file-widget'
            })
        });

        w.render();
        checkWidgetCommon(w);

        w.$('.h-select-file-button').click();
        expect(arg.parentModel).toBe(histomicstk.rootPath);
        arg.onItemClick(item);
        expect(w.model.value()).toBe('model id');

        girder.views.HierarchyWidget = hwidget;
    });

    it('invalid', function () {
        var w = new histomicstk.views.ControlWidget({
            parentView: parentView,
            el: $el.get(0),
            model: new histomicstk.models.Widget({
                type: 'invalid',
                title: 'Title',
                id: 'invalid-widget'
            })
        });
        var _warn = console.warn;
        var message;
        console.warn = function (m) {message = m};

        w.render();
        expect(message).toBe('Invalid widget type "invalid"');
        console.warn = _warn;
    });
});
