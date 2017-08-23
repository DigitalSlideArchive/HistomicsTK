import $ from 'jquery';
import _ from 'underscore';

import { restRequest } from 'girder/rest';
import ItemCollection from 'girder/collections/ItemCollection';
import View from 'girder/views/View';
import 'girder/utilities/jquery/girderModal';

import events from '../events';
import router from '../router';

import template from '../templates/dialogs/openAnnotatedImage.pug';
import '../stylesheets/dialogs/openAnnotatedImage.styl';

let dialog;
let paths = {};

const OpenAnnotatedImage = View.extend({
    events: {
        'click .h-annotated-image': '_submit'
    },

    initialize() {
        this.collection = new ItemCollection();
        this.listenTo(this.collection, 'reset', this.render);
    },

    render() {
        this.$el.html(template({
            items: this.collection.toJSON(),
            paths
        })).girderModal(this);
        this.$el.tooltip();
        return this;
    },

    fetch() {
        var items;
        return restRequest({
            url: 'annotation/images',
            data: {
                limit: 10
            }
        }).then((_items) => {
            items = _items;
            const promises = _.map(items, (item) => {
                return this._getResourcePath(item);
            });
            return $.when(...promises);
        }).then(() => {
            this.collection.reset(items);
            return items;
        });
    },

    _submit(evt) {
        const id = this.$(evt.currentTarget).data('id');
        router.setQuery('bounds', null, {trigger: false});
        router.setQuery('image', id, {trigger: true});
        this.$el.modal('hide');
    },

    _getResourcePath(item) {
        if (_.has(paths, item._id)) {
            return $.Deferred().resolve(paths[item._id]).promise();
        }

        return restRequest({
            url: `resource/${item._id}/path`,
            data: {
                type: 'item'
            }
        }).done((path) => {
            paths[item._id] = path;
        });
    }
});

function createDialog() {
    return new OpenAnnotatedImage({
        parentView: null
    });
}

events.on('h:openAnnotatedImageUi', function () {
    if (!dialog) {
        dialog = createDialog();
    }
    dialog.setElement($('#g-dialog-container')).fetch();
});

export default createDialog;
