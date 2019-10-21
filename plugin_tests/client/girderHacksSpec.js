girderTest.importPlugin('jobs');
girderTest.importPlugin('worker');
girderTest.importPlugin('large_image');
girderTest.importPlugin('slicer_cli_web');
girderTest.importPlugin('HistomicsTK');
girderTest.addScript('/plugins/HistomicsTK/plugin_tests/client/common.js');

girderTest.startApp();

describe('itemList', function () {
    it('login', function () {
        girderTest.login('admin', 'Admin', 'Admin', 'password')();
    });
    it('go to first public user item', function () {
        runs(function () {
            $("a.g-nav-link[g-target='users']").click();
        });
        waitsFor(function () {
            return $('a.g-user-link').length > 0;
        });
        runs(function () {
            $('a.g-user-link').last().click();
        });
        waitsFor(function () {
            return $('a.g-folder-list-link').length > 0;
        });
        runs(function () {
            $('.g-folder-list-link:contains("Public")').click();
        });
        waitsFor(function () {
            return $('a.g-item-list-link:contains("image")').length > 0;
        });
        runs(function () {
            $('a.g-item-list-link:contains("image")').click();
        });
        girderTest.waitForLoad();
        waitsFor(function () {
            return $('.g-item-actions-button').length > 0;
        });
        runs(function () {
            $('.g-item-actions-button').parent().addClass('group');
        });
    });
    it('has a Open HistomicsTK button', function () {
        runs(function () {
            expect($('.g-histomicstk-open-item').length).toBe(1);
        });
    });
    it('has in Quarantine Item button', function () {
        runs(function () {
            expect($('.g-histomicstk-quarantine-item').length).toBe(1);
        });
    });
});
