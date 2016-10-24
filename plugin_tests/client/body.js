if (!String.prototype.startsWith) {
    String.prototype.startsWith = function(searchString, position){
      position = position || 0;
      return this.substr(position, searchString.length) === searchString;
  };
}

window.histomicstk = {};
window.slicer = {};
_.each([
    '/plugins/HistomicsTK/node_modules/sinon/pkg/sinon.js',
    '/clients/web/static/built/plugins/large_image/geo.min.js'
], function (src) {
    $('<script/>', {src: src}).appendTo('head');
});

girderTest.addCoveredScripts([
    '/clients/web/static/built/plugins/jobs/plugin.min.js',
    '/clients/web/static/built/plugins/large_image/plugin.min.js',
    '/clients/web/static/built/plugins/slicer_cli_web/plugin.min.js',
    '/plugins/HistomicsTK/web_client/js/0init.js',
    '/plugins/HistomicsTK/web_client/js/app.js',
    '/plugins/HistomicsTK/web_client/js/views/body.js',
    '/plugins/HistomicsTK/web_client/js/views/header.js',
    '/plugins/HistomicsTK/web_client/js/views/visualization.js',
    '/plugins/HistomicsTK/web_client/js/views/annotationSelectorWidget.js',
    '/clients/web/static/built/plugins/HistomicsTK/templates.js'
]);


girderTest.importStylesheet(
    '/clients/web/static/built/plugins/HistomicsTK/plugin.min.css'
);

describe('routing', function () {
    var rest, $el, setAnalysis, reset;
    beforeEach(function () {
        setAnalysis = sinon.stub(slicer.views.PanelGroup.prototype, 'setAnalysis')
            .returns($.when({}));
        reset = sinon.stub(slicer.views.PanelGroup.prototype, 'reset');
        girder.eventStream = new girder.EventStream();
        rest = sinon.stub(girder, 'restRequest');
        $el = $('<div/>').css({
            width: '500px',
            height: '500px'
        }).attr('id', 'h-body-container').appendTo('body');
    });
    afterEach(function () {
        setAnalysis.restore();
        reset.restore();
        rest.restore();
        $el.remove();
    });
    it('query:analysis', function () {
        rest.onCall(0).returns($.when(''));

        new histomicstk.views.Body({
            parentView: null,
            el: $el
        }).render();

        histomicstk.events.trigger('query:analysis', 'test');
        expect(setAnalysis.callCount).toBe(1);
        expect(reset.callCount).toBe(0);
        expect(setAnalysis.getCall(0).args).toEqual(['test']);

        histomicstk.events.trigger('query:analysis', '');
        expect(setAnalysis.callCount).toBe(1);
        expect(reset.callCount).toBe(1);
    });
    it('visualization change', function () {
        rest.onCall(0).returns($.when(''));

        var body = new histomicstk.views.Body({
            parentView: null,
            el: $el
        });
        body.render();

        sinon.spy(body, '_setImage');
        histomicstk.dialogs.image.model.trigger('change', null);
        expect(body._setImage.callCount).toBe(0);

        var model = new Backbone.Model({
            value: new Backbone.Model({id: 'test'})
        });
        histomicstk.dialogs.image.model.trigger('change', model);
        expect(body._setImage.callCount).toBe(1);
        expect(body._setImage.getCall(0).args).toEqual([model.get('value')]);
    });
});
