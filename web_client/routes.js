import ItemModel from 'girder/models/ItemModel';

import events from './events';
import FrontPageView from './views/body/FrontPageView';
import ImageView from './views/body/ImageView';

import Router from './router';

Router.route('', 'index', function () {
    events.trigger('g:navigateTo', FrontPageView);
});

Router.route('image/:id', 'image', function (id) {
    var model = new ItemModel({_id: id});
    events.trigger('g:navigateTo', ImageView, { model });
});

Router.route('image/:id/analysis/:analysis', function (id, analysis) {
    var model = new ItemModel({_id: id});
    events.trigger('g:navigateTo', ImageView, { model, analysis });
});
