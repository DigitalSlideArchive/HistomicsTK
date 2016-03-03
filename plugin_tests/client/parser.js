$(function () {
    window.histomicstk = {};
    girderTest.addCoveredScripts([
        '/plugins/HistomicsTK/web_client/js/schema/parser.js'
    ]);
});

describe('XML Schema parser', function () {
    describe('constraints', function () {
        it('empty', function () {
            expect(histomicstk.schema._parseConstraints())
                .toEqual({});
        });
        it('missing step', function () {
            var xml = $.parseXML(
                '<constraints><minimum>1</minimum><maximum>3</maximum></constraints>'
            );
            expect(histomicstk.schema._parseConstraints(
                xml
            )).toEqual({min: '1', max: '3'});
        });
        it('full spec', function () {
            var xml = $.parseXML(
                '<constraints><minimum>0</minimum><maximum>2</maximum><step>0.5</step></constraints>'
            );
            expect(histomicstk.schema._parseConstraints(
                xml
            )).toEqual({min: '0', max: '2', step: '0.5'});
        });
    });
    describe('scalar parameter', function () {
        it('basic spec', function () {
            var xml = $.parseXML(
                '<integer>' +
                '<longflag>foo</longflag>' +
                '<label>arg1</label>' +
                '<description>An integer</description>' +
                '</integer>'
            );
            expect(histomicstk.schema._parseScalarParam(
                'integer', xml
            )).toEqual({
                type: 'number',
                slicerType: 'integer',
                id: 'foo',
                title: 'arg1',
                description: 'An integer'
            });
        });
    });
});
